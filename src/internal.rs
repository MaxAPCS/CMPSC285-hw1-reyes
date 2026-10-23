use crate::shaders;
use sample::*;

use image::{ImageFormat, buffer::ConvertBuffer, error::ParameterError};
use nalgebra::{
    Matrix3, Point2, Point3, Projective3, SimdPartialOrd, UnitVector3, Vector2, Vector3, Vector4,
};
use ndarray::{Array2, Array3, ArrayView, Axis, Zip};
use std::debug_assert_matches;
use std::io::Write;
use std::mem::MaybeUninit;

const BLACK: Vector3<f32> = Vector3::new(0., 0., 0.);
const WHITE: Vector3<f32> = Vector3::new(1., 1., 1.);

static BARYCENTRIC_BORDER_THRESHOLD: f32 = 1e-4;
static BARYCENTRIC_DETERMINANT_THRESHOLD: f32 = 1e-12;

/// Running State
/// local to current primitive invocation
#[derive(Clone)]
pub(crate) struct LocalAttributes {
    pub color: Vector3<f32>,
    pub opacity: Vector3<f32>,
    transform: Vec<Projective3<f32>>,
    surface: shaders::RtShaderFn,
    surface_params: Option<Box<[u8]>>,
    displace: shaders::RtShaderFn,
    displace_params: Option<Box<[u8]>>,
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
    pub fn transform_last_cofactor(&self) -> Option<(&Projective3<f32>, Matrix3<f32>)> {
        self.transform.last().map(|t| {
            (
                t,
                Matrix3::new(
                    t[(1, 1)] * t[(2, 2)] - t[(1, 2)] * t[(2, 1)],
                    t[(1, 2)] * t[(2, 0)] - t[(1, 0)] * t[(2, 2)],
                    t[(1, 0)] * t[(2, 1)] - t[(1, 1)] * t[(2, 0)],
                    t[(0, 2)] * t[(2, 1)] - t[(0, 1)] * t[(2, 2)],
                    t[(0, 0)] * t[(2, 2)] - t[(0, 2)] * t[(2, 0)],
                    t[(0, 1)] * t[(2, 0)] - t[(0, 0)] * t[(2, 1)],
                    t[(0, 1)] * t[(1, 2)] - t[(0, 2)] * t[(1, 1)],
                    t[(0, 2)] * t[(1, 0)] - t[(0, 0)] * t[(1, 2)],
                    t[(0, 0)] * t[(1, 1)] - t[(0, 1)] * t[(1, 0)],
                ),
            )
        })
    }

    #[inline]
    pub fn transform_last_mut(&mut self) -> Option<&mut Projective3<f32>> {
        self.transform.last_mut()
    }

    #[inline]
    pub fn set_surface(&mut self, shader: shaders::RtShaderFn, params: Option<Box<[u8]>>) {
        self.surface = shader;
        self.surface_params = params;
    }

    #[inline]
    pub fn set_displace(&mut self, shader: shaders::RtShaderFn, params: Option<Box<[u8]>>) {
        self.displace = shader;
        self.displace_params = params;
    }

    /// World Space In & Out
    pub fn shade_pass(&self, mut mvs: Array2<MicroVertex>) -> Array2<MicroVertex> {
        let mvpos = if let Some(displace) = self.displace {
            // Displacement Shader
            mvs.par_map_inplace(|mv| mv.displace(displace, self.displace_params.as_deref()));
            Some(mvs.map(|mv| mv.wpos))
        } else {
            None
        };

        Zip::indexed(&mut mvs).par_for_each(|(i, j), mv| {
            // Recompute Normals
            if let Some(mvpos) = &mvpos
                && let Some(wnorm) = UnitVector3::try_new(
                    Vector3::cross(
                        &if j == 0 {
                            mv.wpos - mvpos[(i, j + 1)]
                        } else {
                            mvpos[(i, j - 1)] - mv.wpos
                        },
                        &if i == 0 {
                            mv.wpos - mvpos[(i + 1, j)]
                        } else {
                            mvpos[(i - 1, j)] - mv.wpos
                        },
                    ),
                    f32::EPSILON,
                )
            {
                mv.wnorm = wnorm;
            }

            // Surface Shader
            if mv.wnorm.dot(&mv.wpos.coords) > 0. {
                mv.wnorm = -mv.wnorm;
            }
            mv.surface(self.surface, self.surface_params.as_deref());
        });

        mvs
    }
}

/// Global State
/// constant & applicable for entire scene
/// changing width, hight, samples between init_sample_buffer and create_depth/image is UB
pub(crate) struct GlobalOptions {
    pub display: String,
    pub depth_display: bool, /* RiDisplay(name, "zfile", "z") */
    pub width: usize,
    pub hight: usize,
    pub samples: (usize, usize),

    sample_buffer: Option<Array2<SampleBin>>,
    projection: Option<Projective3<f32>>,
    proj_inv_z: nalgebra::Vector4<f32>,
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
            proj_inv_z: nalgebra::Vector4::new(0., 0., 0., 0.),
        }
    }

    /// World Space In, Clip Space Out
    pub fn project_bounds(&self, wpos: &[Point3<f32>]) -> Option<(Point2<f32>, Point2<f32>)> {
        let p = self
            .projection
            .expect("project_bounds called without projection");
        GlobalOptions::bounds_2d(wpos.into_iter().map(|wpos| p.transform_point(wpos).xy()))
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

    /// World Space In
    pub fn project_raster_pass(&mut self, mut mvs: Array2<MicroVertex>) {
        // Projection Pass: Output Clip Space
        let p = self
            .projection
            .expect("project_raster_pass called without projection");
        mvs.par_map_inplace(|mv| {
            let clpos = p.matrix() * mv.wpos.to_homogeneous();
            mv.wpos = Point3::new(
                clpos.x / clpos.w,
                clpos.y / clpos.w,
                self.proj_inv_z.dot(&clpos),
            );
        });

        let sbuf = self
            .sample_buffer
            .as_mut()
            .expect("project_raster_pass called without sample buffer");
        let (hight, width, ..) = sbuf.dim();
        let (hightf, widthf) = (hight as f32, width as f32);
        let clip_to_sample = |clpos: &Point2<f32>, roundup: bool| -> Point2<isize> {
            let x = clpos
                .x
                .algebraic_add(1.)
                .algebraic_mul(widthf.algebraic_div(2.));
            let y = (-clpos.y)
                .algebraic_add(1.)
                .algebraic_mul(hightf.algebraic_div(2.));
            if roundup {
                Point2::new(x.ceil() as _, y.floor() as _)
            } else {
                Point2::new(x.floor() as _, y.ceil() as _)
            }
        };
        let sample_to_clip = |ssx: usize, ssy: usize| -> Point2<f32> {
            Point2::new(
                ((ssx * 2) as f32).algebraic_div(widthf).algebraic_sub(1.),
                (((ssy + 1) * 2) as f32)
                    .algebraic_div(-hightf)
                    .algebraic_add(1.),
            )
        };

        // Sampling Pass
        #[cfg(not(feature = "splat"))]
        mvs.windows((2, 2))
            .into_iter()
            .flat_map(|mvs| {
                std::iter::chain(
                    try_barycentric_passthrough([mvs[(0, 0)], mvs[(1, 0)], mvs[(1, 1)]]),
                    try_barycentric_passthrough([mvs[(0, 0)], mvs[(0, 1)], mvs[(1, 1)]]),
                )
            })
            .for_each(|(mp, bary)| {
                let (csmin, csmax) = &GlobalOptions::bounds_2d(mp.map(|mv| mv.wpos.xy())).unwrap();
                let ssmin = clip_to_sample(csmin, false);
                let ssmax = clip_to_sample(csmax, true);

                if ssmax.x < 0 || ssmin.x >= width as _ || ssmin.y < 0 || ssmax.y >= hight as _ {
                    return;
                }
                let ssy0 = ssmax.y.max(0) as usize;
                let ssx0 = ssmin.x.max(0) as usize;
                Zip::indexed(sbuf.slice_mut(ndarray::s![
                    ssy0..=((ssmin.y as usize).min(hight - 1)),
                    ssx0..=((ssmax.x as usize).min(width - 1)),
                ]))
                .for_each(|(yi, xi), sbin| {
                    let cspos = sample_to_clip(ssx0 + xi, ssy0 + yi) + sbin.jitter();
                    let barys = bary * cspos.to_homogeneous();
                    if barys.iter().any(|&b| {
                        b < -BARYCENTRIC_BORDER_THRESHOLD || b > 1. + BARYCENTRIC_BORDER_THRESHOLD
                    }) {
                        // TODO negotiate edges
                        return;
                    }
                    let cdepth = trilerp(barys, mp.map(|v| v.wpos.z));
                    if cdepth < 0.01 {
                        return;
                    }
                    sbin.insert(Sample {
                        cdepth,
                        color: trilerp(barys, mp.map(|v| v.color)),
                        alpha: trilerp(barys, mp.map(|v| v.alpha)),
                    });
                });
            });
        // Splatting Pass
        #[cfg(feature = "splat")]
        mvs.for_each(|mv| {
            let sspos = clip_to_sample(&mv.wpos.xy(), false);
            let ssx = sspos.x as usize;
            let ssy = sspos.y as usize;
            if 0 <= sspos.x && ssx < width && 0 <= sspos.y && ssy < hight {
                sbuf[(ssy, ssx)].insert(Sample {
                    cdepth: mv.wpos.z,
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
                let mut minz = f32::INFINITY;
                for s in samples {
                    if let Some(sample) = s.get_collapsed()
                        && minz > sample.cdepth
                    {
                        minz = sample.cdepth;
                    }
                }
                minz
            }),
        )
    }

    pub fn create_image(&mut self) -> Option<image::Rgba32FImage> {
        let mut fbuf = Array3::uninit((self.hight, self.width, 4));

        let mut sbuf = self.sample_buffer.take()?;
        sbuf.par_map_inplace(SampleBin::collapse);
        Zip::from(sbuf.exact_chunks_mut(self.samples))
            .and(fbuf.lanes_mut(Axis(2)))
            .par_for_each(|samples, mut lane| {
                let mut color = Vector3::zeros();
                let mut alpha = Vector3::zeros();
                for s in samples {
                    if let Some(s) = s.get_collapsed() {
                        color += s.color;
                        alpha += s.alpha;
                    }
                }
                color.component_div_assign(&alpha);
                alpha /= (self.samples.0 * self.samples.1) as f32;
                lane.assign(&ArrayView::from(
                    &[color.x, color.y, color.z, alpha.sum() / 3.].map(MaybeUninit::new),
                ));
            });

        let (fbuf, off) = unsafe { fbuf.assume_init() }.into_raw_vec_and_offset();
        debug_assert_matches!(off, Some(0) | None);
        image::Rgba32FImage::from_raw(self.width as _, self.hight as _, fbuf)
    }
}

/// Set a projection as the current transformation
#[inline]
pub(crate) fn projection_start(
    lattr: &mut LocalAttributes,
    gopts: &mut GlobalOptions,
    (projection, proj_inv_z): (Projective3<f32>, Vector4<f32>),
) {
    // debug_assert_eq!(projection.inverse().matrix().row(2).transpose(), proj_inv_z); commented out bcz of rounding errors

    // FIXME ideally camera-world transform should be separate
    gopts.projection = Some(projection); // None
    gopts.proj_inv_z = proj_inv_z;
    lattr.transform = vec![Projective3::identity()]; // vec![projection];
}

/// Save the current transformation as projection
#[inline]
pub(crate) fn projection_end(lattr: &mut LocalAttributes, gopts: &mut GlobalOptions) {
    // FIXME ideally camera-world transform should be separate
    // gopts.projection = lattr.transform.pop();
    // lattr.transform = vec![Projective3::identity()];
    let _ = (lattr, gopts);
}

/// Write the depth image to renders/<RiDisplay name> in the handout's
/// format: "Rz01", int32 width and height, then one float per pixel, top
/// row first, all little-endian (the camera-space z of the nearest surface
/// at least half opaque, +infinity where there is none).
pub(crate) fn write_depth(depth: Array2<f32>, path: &str) -> Result<(), std::io::Error> {
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

pub(crate) fn write_image(image: image::Rgba32FImage, path: &str) -> Result<(), image::ImageError> {
    let path = format!("renders/{}", path);
    let Ok(extension) = image::ImageFormat::from_path(&path) else {
        return Err(image::ImageError::Parameter(ParameterError::from_kind(
            image::error::ParameterErrorKind::Generic(format!(
                "write_image called with missing/invalid extension {}",
                path
            )),
        )));
    };
    match extension {
        ImageFormat::Png | ImageFormat::Tga => {
            let rgba: image::RgbaImage = image.convert();
            rgba.save_with_format(&path, extension)
        }
        ImageFormat::Bmp => {
            let rgb: image::RgbImage = image.convert();
            rgb.save_with_format(&path, extension)
        }
        ImageFormat::Jpeg => std::fs::File::create(&path)
            .map_err(image::ImageError::IoError)
            .and_then(|file| {
                let mut out = std::io::BufWriter::new(file);
                image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 95)
                    .encode_image::<image::RgbImage>(&image.convert())
            }),
        _ => {
            let hint = image::error::ImageFormatHint::Exact(extension);
            Err(image::ImageError::Unsupported(
                image::error::UnsupportedError::from_format_and_kind(
                    hint.clone(),
                    image::error::UnsupportedErrorKind::Format(hint),
                ),
            ))
        }
    }
}

/// MicroPolygon Vertex
/// Only contains properties requiring interpolation
#[derive(Clone, Copy)]
pub(crate) struct MicroVertex {
    pub opos: Point3<f32>,       /* object space position */
    pub wpos: Point3<f32>,       /* world space position */
    pub wnorm: UnitVector3<f32>, /* world space normal */
    pub uv: (f32, f32),          /* surface parameters in [0,1] */
    pub color: Vector3<f32>,     /* color */
    pub alpha: Vector3<f32>,     /* opacity */
}
impl Into<shaders::RtShaderContext> for &mut MicroVertex {
    fn into(self) -> shaders::RtShaderContext {
        shaders::RtShaderContext {
            P: self.wpos,
            Po: self.opos,
            N: self.wnorm,
            u: self.uv.0,
            v: self.uv.1,
            Cs: self.color,
            Os: self.alpha,
            params: std::ptr::null(),
            Ci: self.color,
            Oi: self.alpha,
        }
    }
}
impl MicroVertex {
    fn surface(&mut self, shader: shaders::RtShaderFn, params: Option<&[u8]>) {
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
        params: Option<&[u8]>,
    ) {
        let mut ctx: shaders::RtShaderContext = self.into();
        if let Some(params) = params {
            ctx.params = params.as_ptr() as _;
        }
        unsafe { shader(std::ptr::from_mut(&mut ctx)) }
        self.wpos = ctx.P;
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
            self.samples.reserve(4); // avoid future reallocation
            self.samples.push(sample);
        }

        pub fn collapse_depth(&mut self) {
            for sample in &self.samples {
                if sample.alpha.sum() > 0.5 * 3. {
                    let cdepth = sample.cdepth;
                    self.samples.clear();
                    self.samples.push(Sample {
                        cdepth,
                        color: BLACK,
                        alpha: BLACK,
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
            let mut opacity = BLACK;
            let mut color = BLACK;
            for sample in &self.samples {
                if sample.alpha == BLACK {
                    continue;
                }
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
                color: color.simd_clamp(BLACK, WHITE),
                alpha: opacity.simd_clamp(BLACK, WHITE),
            });
        }

        /// Fetch collapsed value
        pub fn get_collapsed(&mut self) -> Option<Sample> {
            debug_assert!(self.samples.len() <= 1);
            debug_assert!(if let Some(sample) = self.samples.last() {
                sample.cdepth == 0. || (sample.color == BLACK && sample.alpha == BLACK)
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
fn try_barycentric_passthrough(vs: [MicroVertex; 3]) -> Option<([MicroVertex; 3], Matrix3<f32>)> {
    try_barycentric(&std::array::from_fn(|i| vs[i].wpos.xy())).map(|bary| (vs, bary))
}

#[inline]
fn try_barycentric(vs: &[Point2<f32>; 3]) -> Option<Matrix3<f32>> {
    let [a, b, c] = vs;
    if a == b || b == c || a == c {
        return None;
    }
    let denom = a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y);
    if denom.abs() < BARYCENTRIC_DETERMINANT_THRESHOLD {
        return None;
    }
    Some(
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
        ) / denom,
    )
}

#[inline]
pub(crate) fn trilerp<T: Copy + std::ops::Sub<Output = T> + std::ops::Add<Output = T>>(
    weights: Vector3<f32>,
    values: [T; 3],
) -> T
where
    f32: std::ops::Mul<T, Output = T>,
{
    values[0] + weights.y * (values[1] - values[0]) + weights.z * (values[2] - values[0])
}
