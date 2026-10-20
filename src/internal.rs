use crate::shaders;
use sample::*;

use image::buffer::ConvertBuffer;
use image::{DynamicImage, ImageFormat};
use nalgebra::{Matrix3, Point2, Point3, Projective3, UnitVector3, Vector2, Vector3};
use ndarray::parallel::prelude::{IntoParallelRefMutIterator, ParallelIterator};
use ndarray::{Array1, Array2, Array3, Axis, Zip};
use std::debug_assert_matches;
use std::io::Write;
use std::mem::MaybeUninit;
use std::sync::Arc;

const WHITE: Vector3<f32> = Vector3::new(1., 1., 1.);

/// Running state, changes model-to-model
#[derive(Clone)]
pub struct ModelState {
    pub color: Vector3<f32>,
    pub opacity: Vector3<f32>,
    transform: Vec<Projective3<f32>>,
    surface: shaders::RtShaderFn,
    surface_params: Option<Arc<[u8]>>,
    displace: shaders::RtShaderFn,
    displace_params: Option<Arc<[u8]>>,

    pub textures: [Option<DynamicImage>; 8], // FIXME don't reset this in frame
}
impl ModelState {
    #[inline]
    pub const fn default() -> Self {
        Self {
            transform: Vec::new(),
            color: WHITE,
            opacity: WHITE,
            surface: None,
            surface_params: None,
            displace: None,
            displace_params: None,
            textures: [const { None }; 8],
        }
    }

    #[inline]
    pub fn transform_push(&mut self) {
        self.transform.push(
            self.transform
                .last()
                .cloned()
                .unwrap_or(Projective3::identity()),
        );
    }

    #[inline]
    #[must_use]
    pub fn transform_pop(&mut self) -> Result<Projective3<f32>, ()> {
        self.transform.pop().ok_or(())
    }

    #[inline]
    pub fn transform_last(&self) -> Option<&Projective3<f32>> {
        self.transform.last()
    }

    #[inline]
    pub fn transform_last_mut(&mut self) -> Option<&mut Projective3<f32>> {
        self.transform.last_mut()
    }

    #[inline]
    pub fn set_surface(&mut self, shader: shaders::RtShaderFn, params: Option<Arc<[u8]>>) {
        self.surface = shader;
        self.surface_params = params;
    }

    #[inline]
    pub fn set_displace(&mut self, shader: shaders::RtShaderFn, params: Option<Arc<[u8]>>) {
        self.displace = shader;
        self.displace_params = params;
    }

    /// World Space In & Out
    pub fn shade_pass(&self, mut mvs: Array2<MicroVertex>) -> Array2<MicroVertex> {
        // Displacement Shader
        mvs.par_iter_mut()
            .for_each(|mv| mv.displace(self.displace, self.displace_params.clone()));

        Zip::from(mvs.windows((2, 2))).par_map_collect(|mvw| {
            // Recompute Normals
            let mut mv = mvw[(0, 0)].clone();
            let flip = mv.cnorm.dot(&mv.cpos.coords) < 0.;
            if let Some(cnorm) = UnitVector3::try_new(
                (mvw[(1, 0)].cpos - mv.cpos).cross(&(mvw[(0, 1)].cpos - mv.cpos)),
                f32::EPSILON,
            ) {
                mv.cnorm = cnorm;
            };

            // Surface Shader
            if flip {
                mv.cnorm = -mv.cnorm;
            }
            mv.surface(self.surface, self.surface_params.clone());

            mv
        })
    }
}

/// properties of the scene
pub struct RenderState {
    pub display: String,
    pub depth_display: bool, /* RiDisplay(name, "zfile", "z") */
    pub width: usize,
    pub hight: usize,
    pub samples: (usize, usize),

    sample_buffer: Option<Array2<SampleBin>>,
    projection: Option<Projective3<f32>>,
}
impl RenderState {
    #[inline]
    pub const fn default() -> Self {
        Self {
            width: 640,
            hight: 480,
            samples: (2, 2),
            display: String::new(),
            depth_display: false,
            sample_buffer: None,
            projection: None,
        }
    }

    /// Camera Space In, Clip Space Out
    pub fn project_bounds(&self, cpos: &[Point3<f32>]) -> (Point2<f32>, Point2<f32>) {
        let proj = self
            .projection
            .expect("bounds_clipspace called without projection");
        RenderState::bounds_2d(cpos.into_iter().map(|cpos| proj.transform_point(cpos).xy()))
    }

    /// Clip Space In & Out
    fn bounds_2d(cspos: impl IntoIterator<Item = Point2<f32>>) -> (Point2<f32>, Point2<f32>) {
        let mut xmin = f32::MAX;
        let mut xmax = f32::MIN;
        let mut ymin = f32::MAX;
        let mut ymax = f32::MIN;

        for cspos in cspos {
            if xmin > cspos.x {
                xmin = cspos.x;
            }
            if xmax < cspos.x {
                xmax = cspos.x;
            }
            if ymin > cspos.y {
                ymin = cspos.y;
            }
            if ymax < cspos.y {
                ymax = cspos.y;
            }
        }

        (Point2::new(xmin, ymin), Point2::new(xmax, ymax))
    }

    /// Clip Space In, Viewport Space Out
    #[inline]
    pub fn viewport_bounds_diag(&self, clbounds: (Point2<f32>, Point2<f32>)) -> usize {
        let cldiag = clbounds.1 - clbounds.0;
        let x = cldiag.x.algebraic_mul((self.width / 2) as f32);
        let y = cldiag.y.algebraic_mul((self.hight / 2) as f32);
        f32::algebraic_add(x.algebraic_mul(x), y.algebraic_mul(y))
            .sqrt()
            .ceil() as usize
    }

    #[inline]
    pub fn init_sample_buffer(&mut self) {
        let shape = (self.hight * self.samples.0, self.width * self.samples.1);
        let shape_recip = Vector2::new(
            ((shape.1 / 2) as f32).recip(),
            ((shape.0 / 2) as f32).recip(),
        );
        self.sample_buffer = Some(Array2::from_shape_fn(shape, |_| {
            SampleBin::new(&shape_recip)
        }))
    }

    /// Camera Space In
    pub fn project_sample_pass(&mut self, mut mvs: Array2<MicroVertex>) {
        // Projection Pass: Output Clip Space
        let p = self
            .projection
            .expect("project_sample_pass called without projection");
        mvs.par_map_inplace(|mv| mv.cpos = p.transform_point(&mv.cpos));

        let sbuf = self
            .sample_buffer
            .as_mut()
            .expect("project_sample_pass called without sample buffer");
        let (hight, width, ..) = sbuf.dim();
        let (hight, width) = (hight as f32, width as f32);
        let clip_to_sample = |clpos: &Point2<f32>| -> Point2<usize> {
            Point2::new(
                ((clpos.x + 1.) * (width / 2.)).floor() as usize,
                ((1. - clpos.y) * (hight / 2.)).floor() as usize,
            )
        };
        let sample_to_clip = |ssx: usize, ssy: usize| -> Point2<f32> {
            Point2::new(
                (ssx as f32 * 2. / width) - 1.,
                1. - ((ssy as f32 + 1.) * 2. / hight),
            )
        };

        // Sampling Pass
        mvs.windows((2, 2)).into_iter().for_each(|mvs| {
            let (csmin, csmax) = &RenderState::bounds_2d(mvs.flatten().map(|mv| mv.cpos.xy()));
            let ssmin = clip_to_sample(csmin);
            let ssmax = clip_to_sample(csmax);
            let tris = [
                barycentric_passthrough([&mvs[(0, 0)], &mvs[(1, 0)], &mvs[(1, 1)]]),
                barycentric_passthrough([&mvs[(0, 0)], &mvs[(0, 1)], &mvs[(1, 1)]]),
            ];

            for ssx in ssmin.x..=ssmax.x {
                for ssy in ssmax.y..=ssmin.y {
                    let Some(sbin) = sbuf.get_mut((ssy, ssx)) else {
                        continue;
                    };
                    let cspos = sample_to_clip(ssx, ssy) + sbin.jitter();
                    // debug_assert_eq!(
                    //     // TODO fix incorrect bucketing
                    //     clip_to_sample(&cspos),
                    //     Point2::new(ssx, ssy),
                    //     "{}",
                    //     sbin.jitter()
                    // );
                    for (vs, barym) in tris {
                        let barys = barym * cspos.to_homogeneous();
                        if barys.iter().any(|&b| b <= 0. || b >= 1.) {
                            // outside triangle
                            // TODO negotiate edges
                            continue;
                        }
                        let mut cdepth = 0.; // TODO fix interpolation
                        let mut color = Vector3::zeros();
                        let mut alpha = Vector3::zeros();
                        for (&b, v) in std::iter::zip(barys.into_iter(), vs) {
                            cdepth += b * v.cpos.z;
                            color += b * v.color;
                            alpha += b * v.alpha;
                        }
                        sbin.insert(Sample {
                            cdepth,
                            color,
                            alpha,
                        });
                    }
                }
            }
        });
        // mvs.for_each(|mv| {
        //     if let Some(s) = sbuf.get_mut((
        //         (((1. - mv.cpos.y) * (hight / 2.)) as usize),
        //         (((mv.cpos.x + 1.) * (width / 2.)) as usize),
        //     )) {
        //         s.insert(Sample {
        //             cdepth: mv.cpos.z,
        //             color: mv.color,
        //             alpha: mv.alpha,
        //         });
        //     }
        // });
    }

    /// Consumes the sample buffer and generates a depth image
    pub fn create_depth(&mut self) -> Option<Array2<f32>> {
        Some(
            Zip::from(self.sample_buffer.take()?.exact_chunks_mut(self.samples)).par_map_collect(
                |samples| {
                    let mut minz = f32::MAX;
                    for s in samples {
                        while let Some(sample) = s.pop() {
                            if sample.cdepth >= minz {
                                break;
                            }
                            if sample.alpha.sum() > 0.5 * 3. {
                                minz = sample.cdepth
                            }
                        }
                    }
                    minz
                },
            ),
        )
    }

    pub fn create_image(&mut self) -> Option<image::RgbaImage> {
        let mut fbuf = Array3::<u8>::uninit((self.hight, self.width, 4));

        let mut sbuf = self.sample_buffer.take()?;
        sbuf.par_map_inplace(SampleBin::collapse);
        Zip::from(sbuf.exact_chunks_mut(self.samples))
            .and(fbuf.lanes_mut(Axis(2)))
            .par_for_each(|samples, mut lane| {
                let mut color = Array1::from_elem(4, 0.);
                for s in samples {
                    if let Some(s) = s.get_collapsed() {
                        color[0] += s.color.x;
                        color[1] += s.color.y;
                        color[2] += s.color.z;
                        color[3] += s.alpha.sum();
                    }
                }
                color /= (self.samples.0 * self.samples.1) as f32;
                let alpha = color[3] / 3.; // average of 3 opacity colors
                if alpha > 0. {
                    color /= alpha;
                }
                color[3] = alpha;
                lane.assign(
                    (*color.mapv(|v| MaybeUninit::new((v.clamp(0.0, 1.0) * 255.5) as u8))).into(),
                );
            });

        let (fbuf, off) = unsafe { fbuf.assume_init() }.into_raw_vec_and_offset();
        debug_assert_matches!(off, Some(0) | None);
        image::RgbaImage::from_raw(self.width as _, self.hight as _, fbuf)
    }
}

#[inline]
pub fn set_projection(
    mstate: &mut ModelState,
    rstate: &mut RenderState,
    projection: Projective3<f32>,
) {
    rstate.projection = Some(projection);
    mstate.transform = vec![Projective3::identity()];
}

/// Write the depth image to renders/<RiDisplay name> in the handout's
/// format: "Rz01", int32 width and height, then one float per pixel, top
/// row first, all little-endian (the camera-space z of the nearest surface
/// at least half opaque, +infinity where there is none).
pub fn write_depth(depth: &Array2<f32>, path: &str) -> Result<(), std::io::Error> {
    let path = format!("renders/{}", path);
    std::fs::File::create(&path).and_then(|file| {
        let mut out = std::io::BufWriter::new(file);
        out.write_all(b"Rz01")?;
        out.write_all(&(depth.dim().1 as i32).to_le_bytes())?;
        out.write_all(&(depth.dim().0 as i32).to_le_bytes())?;
        for d in depth {
            out.write_all(&d.to_le_bytes())?;
        }
        out.flush()
    })
}

pub fn write_image(rgba: image::RgbaImage, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("renders/{}", path);
    let Ok(extension) = image::ImageFormat::from_path(&path) else {
        return Err(format!("write_image called with missing/invalid extension {}", path).into());
    };
    Ok(match extension {
        ImageFormat::Png | ImageFormat::Tga => rgba.save_with_format(&path, extension)?,
        ImageFormat::Bmp => {
            let rgb: image::RgbImage = rgba.convert();
            rgb.save_with_format(&path, extension)?
        }
        ImageFormat::Jpeg => std::fs::File::create(&path)
            .map_err(image::ImageError::IoError)
            .and_then(|file| {
                let mut out = std::io::BufWriter::new(file);
                image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 95)
                    .encode_image::<image::RgbImage>(&rgba.convert())
            })?,
        _ => {
            return Err(format!(
                "write_image called with invalid format {}: use .png, .bmp, .tga, .jpg",
                path
            )
            .into());
        }
    })
}

/// Everything that needs interpolating
#[derive(Clone)] // for recompute normals, bounding box step
pub struct MicroVertex {
    pub cpos: Point3<f32>,       /* camera space position */
    pub opos: Point3<f32>,       /* object space position */
    pub cnorm: UnitVector3<f32>, /* camera space normal */
    pub uv: (f32, f32),          /* surface parameters in [0,1] */
    pub color: Vector3<f32>,     /* color */
    pub alpha: Vector3<f32>,     /* opacity */
}
impl Into<shaders::RtShaderContext> for &mut MicroVertex {
    fn into(self) -> shaders::RtShaderContext {
        shaders::RtShaderContext {
            P: self.cpos,
            Po: self.opos,
            N: self.cnorm,
            u: self.uv.0,
            v: self.uv.1,
            Cs: self.color,
            Os: self.alpha,
            params: std::ptr::null(),
            Ci: Vector3::zeros(),
            Oi: Vector3::zeros(),
        }
    }
}
impl MicroVertex {
    fn surface(&mut self, shader: shaders::RtShaderFn, params: Option<Arc<[u8]>>) {
        let mut ctx: shaders::RtShaderContext = self.into();
        if let Some(params) = params {
            ctx.params = params.as_ptr() as _;
        }
        unsafe { shader.unwrap_or(shaders::STANDARD)(std::ptr::from_mut(&mut ctx)) }
        self.color = ctx.Ci;
        self.alpha = ctx.Oi;
    }
    fn displace(&mut self, shader: shaders::RtShaderFn, params: Option<Arc<[u8]>>) {
        let Some(shader) = shader else { return };
        let mut ctx: shaders::RtShaderContext = self.into();
        if let Some(params) = params {
            ctx.params = params.as_ptr() as _;
        }
        unsafe { shader(std::ptr::from_mut(&mut ctx)) }
        self.cpos = ctx.P;
    }
}

mod sample {
    use super::*;

    pub struct SampleBin {
        jitter: Vector2<f32>,
        samples: Vec<Sample>,
    }
    impl SampleBin {
        /// x [0,sshight), y [0,sswidth)
        pub fn new(sssize: &Vector2<f32>) -> Self {
            Self {
                jitter: Vector2::new_random().component_mul(sssize),
                samples: Vec::new(),
            }
        }

        pub fn jitter(&self) -> &Vector2<f32> {
            &self.jitter
        }

        // for depth only... FIXME
        pub fn pop(&mut self) -> Option<Sample> {
            self.samples.pop()
        }

        /// Insert a sample
        pub fn insert(&mut self, sample: Sample) {
            for i in 0..self.samples.len() {
                if sample.cdepth < self.samples[i].cdepth {
                    if sample.is_opaque() {
                        self.samples[i] = sample;
                        self.samples.truncate(i + 1);
                    } else {
                        self.samples.insert(i, sample);
                    }
                    return;
                }
                if self.samples[i].is_opaque() {
                    return;
                }
            }
            self.samples.push(sample);
        }

        /// Collapse/sum all samples into one
        pub fn collapse(&mut self) {
            if self.samples.is_empty() {
                return;
            }
            let mut opacity = Vector3::zeros();
            let mut color = Vector3::zeros();
            for sample in &self.samples {
                color += sample
                    .color
                    .component_mul(&sample.alpha)
                    .component_mul(&(WHITE - opacity));
                opacity += sample.alpha.component_mul(&(WHITE - opacity));
                if opacity == WHITE {
                    self.samples.clear();
                    break;
                }
            }
            self.samples.clear();
            self.samples.push(Sample {
                cdepth: 0.,
                color,
                alpha: opacity,
            });
        }

        /// Fetch collapsed value
        pub fn get_collapsed(&mut self) -> Option<Sample> {
            debug_assert!(self.samples.len() <= 1);
            debug_assert_eq!(
                if let Some(sample) = self.samples.last() {
                    sample.cdepth
                } else {
                    0.
                },
                0.
            );
            self.samples.pop()
        }
    }

    pub struct Sample {
        pub cdepth: f32,
        pub color: Vector3<f32>,
        pub alpha: Vector3<f32>,
    }
    impl Sample {
        pub fn is_opaque(&self) -> bool {
            self.alpha == WHITE
        }
    }
}

#[inline]
fn barycentric(vs: [Point2<f32>; 3]) -> Matrix3<f32> {
    let [a, b, c] = vs;
    Matrix3::new(
        b.y - c.y,
        c.x - b.x,
        b.x * c.y - c.x * b.y,
        c.y - a.y,
        a.x - c.x,
        c.x * a.y - a.x * c.y,
        a.y - b.y,
        b.x - a.x,
        a.x * b.y - b.x * a.y,
    ) / (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y))
}

#[inline]
fn barycentric_passthrough(vs: [&MicroVertex; 3]) -> ([&MicroVertex; 3], Matrix3<f32>) {
    (vs, barycentric(vs.map(|v| v.cpos.xy())))
}
