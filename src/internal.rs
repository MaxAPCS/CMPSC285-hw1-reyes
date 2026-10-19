use crate::shaders;

use image::buffer::ConvertBuffer;
use image::{DynamicImage, ImageFormat};
use nalgebra::{Point3, Projective3, UnitVector3, Vector2, Vector3};
use ndarray::parallel::prelude::{IntoParallelRefMutIterator, ParallelIterator};
use ndarray::{Array1, Array2, Array3, Axis, Zip, array};
use std::debug_assert_matches;
use std::io::Write;
use std::sync::Arc;
use std::{collections::BinaryHeap, mem::MaybeUninit};

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

    projection: Option<Projective3<f32>>,
    pub textures: [Option<DynamicImage>; 8],
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
            projection: None,
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
    pub fn set_projection(&mut self, projection: Projective3<f32>) {
        self.projection = Some(projection);
        self.transform = vec![Projective3::identity()];
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

    #[inline]
    pub fn bounds_clipspace(&self, opos: &[Point3<f32>]) -> Vector2<f32> {
        let t = self
            .transform_last()
            .expect("bounds_clipspace called without transform");
        let proj = self
            .projection
            .expect("bounds_clipspace called without projection");

        let mut pcc = Array2::uninit((opos.len(), 2));
        Zip::from(opos)
            .and(pcc.lanes_mut(Axis(1)))
            .par_for_each(|p, mut pcc| {
                let p = proj.transform_point(&t.transform_point(p)).xy();
                pcc.assign(&array![MaybeUninit::new(p.x), MaybeUninit::new(p.y)]);
            });
        let pcc = unsafe { pcc.assume_init() };

        fn range<'a>(iter: impl IntoIterator<Item = &'a f32>) -> (f32, f32) {
            iter.into_iter().fold((f32::MAX, f32::MIN), |mut r, &x| {
                if r.0 > x {
                    r.0 = x
                }
                if r.1 < x {
                    r.1 = x
                }
                r
            })
        }

        let (xmin, xmax) = range(pcc.row(0));
        let (ymin, ymax) = range(pcc.row(1));

        Vector2::new(xmax - xmin, ymax - ymin)
    }

    pub fn shade_project_pass(&self, mut mvs: Array2<MicroVertex>) -> Array2<Option<MicroVertex>> {
        let p = self
            .projection
            .expect("shade_project_pass called without projection");

        // Displacement Shader
        mvs.par_iter_mut()
            .for_each(|mv| mv.displace(self.displace, self.displace_params.clone()));

        Zip::from(mvs.windows((2, 2))).par_map_collect(|mvw| {
            // Recompute Normals
            let mut mv = mvw[(0, 0)].clone();
            if let Some(cnorm) = UnitVector3::try_new(
                (mvw[(1, 0)].cpos - mv.cpos).cross(&(mvw[(0, 1)].cpos - mv.cpos)),
                0.01,
            ) {
                mv.cnorm = cnorm;
            };

            // Surface Shader
            if mv.cnorm.dot(&mv.cpos.coords) < 0. {
                mv.cnorm = -mv.cnorm;
            }
            mv.surface(self.surface, self.surface_params.clone());

            // Projection (cpos is now clip position)
            mv.cpos = p.transform_point(&mv.cpos);
            if mv.cpos.x <= -1.
                || mv.cpos.x > 1.
                || mv.cpos.y <= -1.
                || mv.cpos.y > 1.
                || mv.cpos.z < 0.01
            {
                return None;
            }
            Some(mv)
        })
    }
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

#[derive(Clone, Copy)] // exclusively so populating the Array2 works
struct Sample {
    cdepth: f32,
    color: Vector3<f32>,
    alpha: Vector3<f32>,
}
impl PartialEq for Sample {
    fn eq(&self, other: &Self) -> bool {
        self.cdepth == other.cdepth
    }
}
impl Eq for Sample {}
impl PartialOrd for Sample {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        other.cdepth.partial_cmp(&self.cdepth)
    }
}
impl Ord for Sample {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.cdepth.total_cmp(&self.cdepth)
    }
}

/// properties of the scene
pub struct RenderState {
    pub display: String,
    pub depth_display: bool, /* RiDisplay(name, "zfile", "z") */
    pub width: usize,
    pub hight: usize,
    pub samples: (usize, usize),
    sample_buffer: Option<Array2<BinaryHeap<Sample>>>,
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
        }
    }

    #[inline]
    pub fn diag_pixels_screenspace(&self, ssv: &Vector2<f32>) -> usize {
        let x = (1. - ssv.y).algebraic_mul((self.hight / 2) as f32);
        let y = (ssv.x + 1.).algebraic_mul((self.width / 2) as f32);
        f32::algebraic_add(x.algebraic_mul(x), y.algebraic_mul(y))
            .sqrt()
            .ceil() as usize
    }

    #[inline]
    pub fn init_sample_buffer(&mut self) {
        self.sample_buffer = Some(Array2::from_elem(
            (self.hight * self.samples.0, self.width * self.samples.1),
            BinaryHeap::new(),
        ))
    }

    pub fn sample_pass(&mut self, mvs: Array2<Option<MicroVertex>>) {
        let sbuf = self
            .sample_buffer
            .as_mut()
            .expect("RiSphere called before RiWorldBegin");
        let (hight, width, ..) = sbuf.dim();

        mvs.for_each(|mv| {
            let Some(mv) = mv else { return };
            // TODO: change splatting to real sampling
            sbuf[(
                (((1. - mv.cpos.y) * (hight / 2) as f32) as usize),
                (((mv.cpos.x + 1.) * (width / 2) as f32) as usize),
            )]
                .push(Sample {
                    cdepth: mv.cpos.z,
                    color: mv.color,
                    alpha: mv.alpha,
                }); // TODO: discard if after opaque object
        });
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
        sbuf.par_map_inplace(|s| {
            if s.is_empty() {
                return;
            }
            let mut opacity = Vector3::zeros();
            let mut color = Vector3::zeros();
            while let Some(sample) = s.pop() {
                color += sample
                    .color
                    .component_mul(&sample.alpha)
                    .component_mul(&(WHITE - opacity));
                opacity += sample.alpha.component_mul(&(WHITE - opacity));
                if opacity == WHITE {
                    s.clear();
                    break;
                }
            }
            s.push(Sample {
                cdepth: 0.,
                color,
                alpha: opacity,
            });
        });
        Zip::from(sbuf.exact_chunks_mut(self.samples))
            .and(fbuf.lanes_mut(Axis(2)))
            .par_for_each(|samples, mut lane| {
                let mut color = Array1::from_elem(4, 0.);
                for s in samples {
                    if let Some(s) = s.pop() {
                        color[0] += s.color.x;
                        color[1] += s.color.y;
                        color[2] += s.color.z;
                        color[3] += s.alpha.sum();
                    }
                }
                color /= (self.samples.0 * self.samples.1) as f32;
                color[3] /= 3.; // average of 3 opacity colors
                lane.assign(
                    (*color.mapv(|v| MaybeUninit::new((v.clamp(0.0, 1.0) * 255.0).round() as u8)))
                        .into(),
                );
            });

        let fbuf = unsafe { fbuf.assume_init() };
        let (fbuf, off) = fbuf.into_raw_vec_and_offset();
        debug_assert_matches!(off, Some(0) | None);
        image::RgbaImage::from_raw(self.width as _, self.hight as _, fbuf)
    }
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
