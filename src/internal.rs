use crate::shaders;
use sample::*;

use image::ImageFormat;
use image::buffer::ConvertBuffer;
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
pub struct LocalAttributes {
    pub color: Vector3<f32>,
    pub opacity: Vector3<f32>,
    transform: Vec<Projective3<f32>>,
    surface: shaders::RtShaderFn,
    surface_params: Option<Arc<[u8]>>,
    displace: shaders::RtShaderFn,
    displace_params: Option<Arc<[u8]>>,
}
impl LocalAttributes {
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
        if let Some(displace) = self.displace {
            mvs.par_iter_mut()
                .for_each(|mv| mv.displace(displace, self.displace_params.clone()));
        }

        Zip::from(mvs.windows((2, 2))).par_map_collect(|mvw| {
            // Recompute Normals
            let mut mv = mvw[(0, 0)];
            let flip = mv.cnorm.dot(&mv.cpos.coords) > 0.;
            if self.displace.is_some()
                && let Some(cnorm) = UnitVector3::try_new(
                    (mvw[(1, 0)].cpos - mv.cpos).cross(&(mvw[(0, 1)].cpos - mv.cpos)),
                    f32::EPSILON,
                )
            {
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

/// properties of all frames in scene
pub struct GlobalOptions {
    pub display: String,
    pub depth_display: bool, /* RiDisplay(name, "zfile", "z") */
    pub width: usize,
    pub hight: usize,
    pub samples: (usize, usize),

    sample_buffer: Option<Array2<SampleBin>>,
    projection: Option<Projective3<f32>>,
}
impl GlobalOptions {
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
    pub fn project_bounds(&self, cpos: &[Point3<f32>]) -> Option<(Point2<f32>, Point2<f32>)> {
        let proj = self
            .projection
            .expect("bounds_clipspace called without projection");
        GlobalOptions::bounds_2d(cpos.into_iter().map(|cpos| proj.transform_point(cpos).xy()))
    }

    /// Clip Space In & Out
    fn bounds_2d<T: nalgebra::Scalar + Copy + PartialOrd>(
        cspos: impl IntoIterator<Item = Point2<T>>,
    ) -> Option<(Point2<T>, Point2<T>)> {
        let mut cspos = cspos.into_iter();
        let Some(first) = cspos.next() else {
            return None;
        };
        let mut xmin = first.x;
        let mut xmax = first.x;
        let mut ymin = first.y;
        let mut ymax = first.y;

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

        Some((Point2::new(xmin, ymin), Point2::new(xmax, ymax)))
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
            2f32.algebraic_div(shape.1 as _),
            2f32.algebraic_div(shape.0 as _),
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
        let clip_to_sample = |clpos: &Point2<f32>, roundup: bool| -> Point2<usize> {
            let x = clpos
                .x
                .algebraic_add(1.)
                .algebraic_mul(width.algebraic_div(2.));
            let y = (-clpos.y)
                .algebraic_add(1.)
                .algebraic_mul(hight.algebraic_div(2.));
            if roundup {
                Point2::new(x.ceil() as usize, y.floor() as usize)
            } else {
                Point2::new(x.floor() as usize, y.ceil() as usize)
            }
        };
        let sample_to_clip = |ssx: usize, ssy: usize| -> Point2<f32> {
            Point2::new(
                ((ssx * 2) as f32).algebraic_div(width).algebraic_sub(1.),
                (-((ssy * 2) as f32)).algebraic_div(hight).algebraic_add(1.),
            )
        };

        #[cfg(not(feature = "splat"))]
        // Sampling Pass
        mvs.windows((2, 2))
            .into_iter()
            .flat_map(|mvs| {
                std::iter::chain(
                    barycentric_passthrough([mvs[(0, 0)], mvs[(1, 0)], mvs[(1, 1)]]),
                    barycentric_passthrough([mvs[(0, 0)], mvs[(0, 1)], mvs[(1, 1)]]),
                )
            })
            .for_each(|(mp, bary)| {
                let (csmin, csmax) = &GlobalOptions::bounds_2d(mp.map(|mv| mv.cpos.xy())).unwrap();
                let ssmin = clip_to_sample(csmin, false);
                let ssmax = clip_to_sample(csmax, true);

                for ssx in ssmin.x..=ssmax.x {
                    for ssy in ssmax.y..=ssmin.y {
                        let Some(sbin) = sbuf.get_mut((ssy, ssx)) else {
                            continue;
                        };
                        let cspos = sample_to_clip(ssx, ssy) + sbin.jitter();
                        let barys = bary * cspos.to_homogeneous();
                        if barys.iter().any(|&b| b < 0. || b > 1.) {
                            // outside triangle
                            // TODO negotiate edges
                            continue;
                        }
                        let cdepth = trilerp(barys, mp.map(|v| v.cpos.z));
                        if cdepth < 0.01 {
                            continue;
                        }
                        sbin.insert(Sample {
                            cdepth,
                            color: trilerp(barys, mp.map(|v| v.color)),
                            alpha: trilerp(barys, mp.map(|v| v.alpha)),
                        });
                    }
                }
            });
        #[cfg(feature = "splat")]
        // Splatting
        mvs.for_each(|mv| {
            let sspos = clip_to_sample(&mv.cpos.xy(), false);
            if let Some(sbin) = sbuf.get_mut((sspos.y, sspos.x)) {
                sbin.insert(Sample {
                    cdepth: mv.cpos.z,
                    color: mv.color,
                    alpha: mv.alpha,
                });
            }
        });
    }

    /// Consumes the sample buffer and generates a depth image
    pub fn create_depth(&mut self) -> Option<Array2<f32>> {
        let mut sbuf = self.sample_buffer.take()?;
        sbuf.par_map_inplace(SampleBin::collapse_depth);
        Some(
            Zip::from(sbuf.exact_chunks_mut(self.samples)).par_map_collect(|samples| {
                let mut minz = f32::MAX;
                for s in samples {
                    if let Some(sample) = s.get_collapsed() {
                        if sample.cdepth < minz {
                            minz = sample.cdepth;
                        }
                    }
                }
                minz
            }),
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
    lattr: &mut LocalAttributes,
    gopts: &mut GlobalOptions,
    projection: Projective3<f32>,
) {
    gopts.projection = Some(projection);
    lattr.transform = vec![Projective3::identity()];
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
#[derive(Clone, Copy)]
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
    fn displace(
        &mut self,
        shader: unsafe extern "C" fn(*mut shaders::RtShaderContext),
        params: Option<Arc<[u8]>>,
    ) {
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

        /// Insert a sample
        pub fn insert(&mut self, sample: Sample) {
            for i in 0..self.samples.len() {
                if sample.cdepth <= self.samples[i].cdepth {
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

        pub fn collapse_depth(&mut self) {
            for sample in &self.samples {
                if sample.alpha.sum() > 0.5 * 3. {
                    let cdepth = sample.cdepth;
                    self.samples.clear();
                    self.samples.push(Sample {
                        cdepth,
                        color: Vector3::zeros(),
                        alpha: Vector3::zeros(),
                    });
                    return;
                }
            }
            self.samples.clear();
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
            debug_assert!(if let Some(sample) = self.samples.last() {
                sample.cdepth == 0.
                    || (sample.color == Vector3::zeros() && sample.alpha == Vector3::zeros())
            } else {
                true
            });
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
fn barycentric(vs: &[Point2<f32>; 3]) -> Matrix3<f32> {
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
fn barycentric_passthrough(vs: [MicroVertex; 3]) -> Option<([MicroVertex; 3], Matrix3<f32>)> {
    let clpos = std::array::from_fn(|i| vs[i].cpos.xy());
    if clpos[0] == clpos[1] || clpos[1] == clpos[2] || clpos[0] == clpos[2] {
        None // degenerate
    } else {
        Some((vs, barycentric(&clpos)))
    }
}

#[inline]
fn trilerp<T: Copy + std::ops::Sub<Output = T> + std::ops::Add<Output = T>>(
    weights: Vector3<f32>,
    values: [T; 3],
) -> T
where
    f32: std::ops::Mul<T, Output = T>,
{
    values[0] + weights.y * (values[1] - values[0]) + weights.z * (values[2] - values[0])
}
