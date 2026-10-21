#![allow(non_snake_case)]

use crate::{internal::*, shaders};

use image::DynamicImage;
use nalgebra::{Matrix4, Orthographic3, Perspective3};
use ndarray::Array2;
use std::ffi::{CStr, c_char, c_void};
use std::sync::{Mutex, RwLock};

/* ------------------------------------------------------------------ types */

pub type RtFloat = f32;
pub type RtInt = i32;
pub type RtToken = *const c_char;
pub type RtPointer = *mut c_void;
pub type RtColor = nalgebra::Vector3<RtFloat>; // compatible with [RtFloat; 3]
pub type RtPoint = nalgebra::Point3<RtFloat>; // compatible with [RtFloat; 3]
pub type RtVector = nalgebra::Vector3<RtFloat>; // compatible with [RtFloat; 3]
pub type RtNormal = nalgebra::UnitVector3<RtFloat>; // compatible with [RtFloat; 3]
pub type RtMatrix = nalgebra::Projective3<RtFloat>; // compatible with [[RtFloat; 4]; 4]

/// A C string argument as a Rust &str ("" for a null pointer).
#[inline]
unsafe fn read_token(s: RtToken) -> &'static str {
    if s.is_null() {
        ""
    } else {
        unsafe { CStr::from_ptr(s) }.to_str().unwrap_or("")
    }
}

static DICERATE_MIN: usize = 4;
static DICERATE_MAX: usize = 2048;
static HITHER: RtFloat = 0.1;
static YONDER: RtFloat = 10000.;

static GOPTS: Mutex<GlobalOptions> = Mutex::new(GlobalOptions::default());
static LATTR: RwLock<LocalAttributes> = RwLock::new(LocalAttributes::default());
static FRAME_LATTR: Mutex<Option<LocalAttributes>> = Mutex::new(None);
static TEXTURES: Mutex<[Option<DynamicImage>; 8]> = Mutex::new([const { None }; 8]);

/* ---------------------------------------------------------- graphics state */

/// RiBegin starts a rendering session and resets the graphics state to
/// RenderMan's defaults: color and opacity (1, 1, 1), surface shader
/// STANDARD, no displacement, orthographic projection, 640 x 480 pixels,
/// 2 x 2 samples. Scenes rely on these (FirstPrimitive never calls
/// RiOpacity). RiEnd ends the session (free textures and buffers here).
/// The name is always RI_NULL.
#[unsafe(no_mangle)]
pub extern "C" fn RiBegin(_name: RtToken) {
    let mut gopts = GOPTS.lock().unwrap();
    *gopts = GlobalOptions::default();
    let mut lattr = LATTR.write().unwrap();
    *lattr = LocalAttributes::default();
    let mut flattr = FRAME_LATTR.lock().unwrap();
    *flattr = None;
    let mut textures = TEXTURES.lock().unwrap();
    *textures = [const { None }; 8];
}

#[unsafe(no_mangle)]
pub extern "C" fn RiEnd() {}

/// The image size in pixels. The screen window is [-a, a] x [-1, 1] for
/// a = xres / yres >= 1 and [-1, 1] x [-1/a, 1/a] for a < 1, so the field of
/// view spans the shorter side (Section 2).
#[unsafe(no_mangle)]
pub extern "C" fn RiFormat(xres: RtInt, yres: RtInt) {
    let mut gopts = GOPTS.lock().unwrap();
    gopts.width = xres as usize;
    gopts.hight = yres as usize;
}

/// xsamples x ysamples jittered samples per pixel (Section 4).
#[unsafe(no_mangle)]
pub extern "C" fn RiPixelSamples(xsamples: RtInt, ysamples: RtInt) {
    let mut gopts = GOPTS.lock().unwrap();
    gopts.samples = (xsamples.max(1) as usize, ysamples.max(1) as usize);
}

/// The output file, written into renders/ by RiWorldEnd. RiDisplay(name, "",
/// "") names the picture; RiDisplay(name, "zfile", "z") asks for the depth
/// image instead (Section 4, "Depth output").
#[unsafe(no_mangle)]
pub extern "C" fn RiDisplay(name: *const c_char, type_: RtToken, mode: RtToken) {
    let mut gopts = GOPTS.lock().unwrap();
    gopts.display = unsafe { read_token(name) }.to_string();
    gopts.depth_display = unsafe { read_token(type_) == "zfile" || read_token(mode) == "z" };
}

/// RiFrameBegin/End bracket one frame of an animation: save the graphics
/// state at Begin and restore it at End, so that each frame starts afresh.
#[unsafe(no_mangle)]
pub extern "C" fn RiFrameBegin(_frame: RtInt) {
    let lattr = LATTR.read().unwrap();
    let mut flattr = FRAME_LATTR.lock().unwrap();
    *flattr = Some(lattr.clone());
}

#[unsafe(no_mangle)]
pub extern "C" fn RiFrameEnd() {
    *LATTR.write().unwrap() = FRAME_LATTR
        .lock()
        .unwrap()
        .take()
        .expect("RiFrameEnd called before RiFrameBegin");
}

#[unsafe(no_mangle)]
pub extern "C" fn RiWorldBegin() {
    GOPTS.lock().unwrap().init_sample_buffer();
}

#[unsafe(no_mangle)]
pub extern "C" fn RiWorldEnd() {
    let mut gopts = GOPTS.lock().unwrap();
    if gopts.depth_display {
        write_depth(
            &gopts
                .create_depth()
                .expect("RiWorldEnd called before RiWorldBegin"),
            &gopts.display,
        )
        .unwrap();
    } else {
        write_image(
            gopts
                .create_image()
                .expect("RiWorldEnd called before RiWorldBegin"),
            &gopts.display,
        )
        .unwrap();
    }
}

/* ----------------------------------------------------------- transformations */

/// RiProjection(RI_PERSPECTIVE, "fov", &fov), fov in degrees over the
/// shorter side of the image, or RiProjection(RI_ORTHOGRAPHIC, NULL, NULL):
/// the C++ scenes omit the last two arguments, which default to NULL. It
/// also resets the current transformation to the identity: the camera
/// transform is built after it (Section 2). value points at an RtFloat.
#[unsafe(no_mangle)]
pub extern "C" fn RiProjection(name: RtToken, _token: RtToken, value: RtPointer) {
    let name = unsafe { read_token(name) };

    let mut gopts = GOPTS.lock().unwrap();
    let aspect = (gopts.width as f32) / (gopts.hight as f32);

    let mut lattr = LATTR.write().unwrap();
    set_projection(
        &mut lattr,
        &mut gopts,
        match name {
            "perspective" => {
                let fov = unsafe { *(value as *const _ as *const RtFloat) }.to_radians();
                let mut persp = Perspective3::new(aspect, fov, HITHER, YONDER).to_projective();
                persp
                    .matrix_mut_unchecked()
                    .prepend_nonuniform_scaling_mut(&RtVector::new(1.0, 1.0, -1.0));
                persp
            }
            "orthographic" => if aspect >= 1. {
                Orthographic3::new(-aspect, aspect, -1., 1., YONDER, HITHER)
            } else {
                Orthographic3::new(-1., 1., -aspect.recip(), aspect.recip(), YONDER, HITHER)
            }
            .to_projective(),
            _ => panic!("RiProjection called with invalid name"),
        },
    );
}

#[unsafe(no_mangle)]
pub extern "C" fn RiTransformBegin() {
    LATTR.write().unwrap().transform_push();
}

#[unsafe(no_mangle)]
pub extern "C" fn RiTransformEnd() {
    LATTR
        .write()
        .unwrap()
        .transform_pop()
        .expect("RiTransformEnd called before RiWorldBegin or RtProjection");
}

#[unsafe(no_mangle)]
pub extern "C" fn RiConcatTransform(transform: &[RtFloat; 16]) {
    let transform = RtMatrix::from_matrix_unchecked(Matrix4::from_row_slice(transform));
    *LATTR
        .write()
        .unwrap()
        .transform_last_mut()
        .expect("RiConcatTransform called before RiWorldBegin or RtProjection") *= transform;
}

#[unsafe(no_mangle)]
pub extern "C" fn RiRotate(angle: RtFloat, dx: RtFloat, dy: RtFloat, dz: RtFloat) {
    *LATTR
        .write()
        .unwrap()
        .transform_last_mut()
        .expect("RiTranslate called before RiTransformBegin")
        .matrix_mut_unchecked() *= Matrix4::from_axis_angle(
        &RtNormal::new_normalize(RtVector::new(dx, dy, dz)),
        angle.to_radians(),
    );
}

#[unsafe(no_mangle)]
pub extern "C" fn RiTranslate(dx: RtFloat, dy: RtFloat, dz: RtFloat) {
    LATTR
        .write()
        .unwrap()
        .transform_last_mut()
        .expect("RiTranslate called before RiWorldBegin or RtProjection")
        .matrix_mut_unchecked()
        .prepend_translation_mut(&RtVector::new(dx, dy, dz));
}

#[unsafe(no_mangle)]
pub extern "C" fn RiScale(sx: RtFloat, sy: RtFloat, sz: RtFloat) {
    LATTR
        .write()
        .unwrap()
        .transform_last_mut()
        .expect("RiTranslate called before RiWorldBegin or RtProjection")
        .matrix_mut_unchecked()
        .prepend_nonuniform_scaling_mut(&RtVector::new(sx, sy, sz));
}

/* ------------------------------------------------------------------ shading */

/// Attributes: they stay set until changed (RiTransformEnd does not restore
/// them). The color and opacity become Cs and Os in the shader context.
/// An RtColor argument arrives as a pointer to its three floats.
#[unsafe(no_mangle)]
pub extern "C" fn RiColor(color: &RtColor) {
    LATTR.write().unwrap().color = *color;
}

#[unsafe(no_mangle)]
pub extern "C" fn RiOpacity(opacity: &RtColor) {
    LATTR.write().unwrap().opacity = *opacity;
}

/// Bind a shader to the primitives that follow. Copy nbytes of params now
/// (std::slice::from_raw_parts(params as *const u8, nbytes).to_vec()): the
/// scene's struct may be gone by the time the shader runs. A None shader
/// means STANDARD for RiSurface and no displacement for RiDisplacement
/// (Section 5).
#[unsafe(no_mangle)]
pub extern "C" fn RiSurface(shader: shaders::RtShaderFn, params: *const c_void, nbytes: usize) {
    LATTR.write().unwrap().set_surface(
        shader,
        if !params.is_null() && nbytes > 0 {
            // SAFETY: params null check, u8 is always valid
            let params = unsafe { std::slice::from_raw_parts(params as *const u8, nbytes) };
            Some(params.into())
        } else {
            None
        },
    )
}

#[unsafe(no_mangle)]
pub extern "C" fn RiDisplacement(
    shader: shaders::RtShaderFn,
    params: *const c_void,
    nbytes: usize,
) {
    LATTR.write().unwrap().set_displace(
        shader,
        if !params.is_null() && nbytes > 0 {
            // SAFETY: params null check, u8 is always valid
            let params = unsafe { std::slice::from_raw_parts(params as *const u8, nbytes) };
            Some(params.into())
        } else {
            None
        },
    )
}

/// Load an image into one of 8 texture slots with the image crate
#[unsafe(no_mangle)]
pub extern "C" fn RiMakeTexture(file: *const c_char, slot: RtInt) {
    let path = unsafe { read_token(file) };
    let image = image::open(format!("assets/{}", path))
        .or(image::open(format!("assets/textures/{}", path)))
        .expect("RiMakeTexture called with invalid file");

    TEXTURES.lock().unwrap()[slot as usize] = Some(image);
}

#[inline]
fn wrap(v: f32) -> f32 {
    if v >= 0.0 && v <= 1.0 {
        v
    } else {
        v - v.floor()
    }
}

/// look it up bilinearly with repeat wrap: s runs left to right, t from the
/// TOP row (t = 0) to the bottom (Section 6). An empty slot reads black.
/// result points at three floats.
#[unsafe(no_mangle)]
pub extern "C" fn RiTexture(slot: RtInt, s: RtFloat, t: RtFloat, result: &mut RtVector) {
    let textures = TEXTURES.lock().unwrap();
    let image = textures[slot as usize]
        .as_ref()
        .expect("RiTexture called with invalid slot");

    result.fill(0.);
    if let Some(sample) = image::imageops::sample_bilinear(image, wrap(s), wrap(t)) {
        result[0] = sample.0[0] as f32 / u8::MAX as f32;
        result[1] = sample.0[1] as f32 / u8::MAX as f32;
        result[2] = sample.0[2] as f32 / u8::MAX as f32;
    }
}

/* ---------------------------------------------------------------- primitives */

// Each primitive goes through the REYES pipeline as soon as it is called:
// bound it, choose a dice rate from the bound's size on screen, dice it
// into a grid of micropolygons, shade the grid, and hand the micropolygons
// to the hider (Section 3). The quadrics use the RenderMan parameterizations
// (RiSpec section 5.4); thetamax, phimin and phimax are degrees.
//
// Shade the grid in two passes (Section 5): the displacement shader at
// every vertex, then N recomputed from the displaced grid, then the surface
// shader at every vertex. Each call fills one vertex's RtShaderContext,
// runs the shader and keeps what it wrote. A shader sees only its own
// vertex, so the order you visit them in does not matter; only the new N
// needs the neighbors, which is why it waits for the whole first pass.

macro_rules! make_corners {
    ($x:expr, $y:expr, $z1:expr, $z2:expr) => {
        [
            RtPoint::new($x, $y, $z1),
            RtPoint::new(-$x, $y, $z1),
            RtPoint::new($x, -$y, $z1),
            RtPoint::new(-$x, -$y, $z1),
            RtPoint::new($x, $y, $z2),
            RtPoint::new(-$x, $y, $z2),
            RtPoint::new($x, -$y, $z2),
            RtPoint::new(-$x, -$y, $z2),
        ]
    };
}

#[inline]
fn linear_cofactor(m: Matrix4<f32>) -> nalgebra::Matrix3<f32> {
    nalgebra::Matrix3::new(
        m[(1, 1)] * m[(2, 2)] - m[(1, 2)] * m[(2, 1)],
        m[(1, 2)] * m[(2, 0)] - m[(1, 0)] * m[(2, 2)],
        m[(1, 0)] * m[(2, 1)] - m[(1, 1)] * m[(2, 0)],
        m[(0, 2)] * m[(2, 1)] - m[(0, 1)] * m[(2, 2)],
        m[(0, 0)] * m[(2, 2)] - m[(0, 2)] * m[(2, 0)],
        m[(0, 1)] * m[(2, 0)] - m[(0, 0)] * m[(2, 1)],
        m[(0, 1)] * m[(1, 2)] - m[(0, 2)] * m[(1, 1)],
        m[(0, 2)] * m[(1, 0)] - m[(0, 0)] * m[(1, 2)],
        m[(0, 0)] * m[(1, 1)] - m[(0, 1)] * m[(1, 0)],
    )
}

#[unsafe(no_mangle)]
pub extern "C" fn RiSphere(radius: RtFloat, zmin: RtFloat, zmax: RtFloat, thetamax: RtFloat) {
    let mut gopts = GOPTS.lock().unwrap();
    let lattr = LATTR.read().unwrap();

    let t = lattr
        .transform_last()
        .expect("RiSphere called before RiTransformBegin");
    let tdual = linear_cofactor(*t.matrix());

    let dicerate = (4 * gopts.viewport_bounds_diag(
        gopts
            .project_bounds(
                &make_corners!(radius, radius, zmin, zmax).map(|p| t.transform_point(&p)),
            )
            .unwrap(),
    ))
    .clamp(DICERATE_MIN, DICERATE_MAX);

    let ther = thetamax.to_radians();
    let phis = (zmin / radius).clamp(-1., 1.).asin();
    let phir = (zmax / radius).clamp(-1., 1.).asin().algebraic_sub(phis);

    let mut mvs = Array2::uninit((dicerate + 2, dicerate + 1));
    for i in 0..mvs.dim().0 {
        let u = i as f32 / dicerate as f32;
        let (sinth, costh) = ther.algebraic_mul(u).sin_cos();
        for j in 0..mvs.dim().1 {
            let v = j as f32 / dicerate as f32;
            let (sinph, cosph) = phis.algebraic_add(phir.algebraic_mul(v)).sin_cos();

            let norm = RtVector::new(costh * cosph, sinth * cosph, sinph);
            let opos = RtPoint::from(norm.scale(radius));
            mvs[(i, j)].write(MicroVertex {
                cpos: t.transform_point(&opos),
                opos,
                cnorm: RtNormal::new_normalize(tdual * norm),
                uv: (u, v),
                color: lattr.color,
                alpha: lattr.opacity,
            });
        }
    }

    gopts.project_sample_pass(lattr.shade_pass(unsafe { mvs.assume_init() }));
}

#[unsafe(no_mangle)]
pub extern "C" fn RiCylinder(radius: RtFloat, zmin: RtFloat, zmax: RtFloat, thetamax: RtFloat) {
    let mut gopts = GOPTS.lock().unwrap();
    let lattr = LATTR.read().unwrap();

    let t = lattr
        .transform_last()
        .expect("RiCylinder called before RiTransformBegin");
    let tdual = linear_cofactor(*t.matrix());

    let dicerate = (4 * gopts.viewport_bounds_diag(
        gopts
            .project_bounds(
                &make_corners!(radius, radius, zmin, zmax).map(|p| t.transform_point(&p)),
            )
            .unwrap(),
    ))
    .clamp(DICERATE_MIN, DICERATE_MAX);

    let ther = thetamax.to_radians();

    let mut mvs = Array2::uninit((dicerate + 2, dicerate + 1));
    for i in 0..mvs.dim().0 {
        let u = i as f32 / dicerate as f32;
        let (sinth, costh) = ther.algebraic_mul(u).sin_cos();
        for j in 0..mvs.dim().1 {
            let v = j as f32 / dicerate as f32;

            let onorm = RtVector::new(costh, sinth, 0.);
            let opos = RtPoint::new(radius * costh, radius * sinth, zmin + v * (zmax - zmin));
            mvs[(i, j)].write(MicroVertex {
                cpos: t.transform_point(&opos),
                opos,
                cnorm: RtNormal::new_normalize(tdual * onorm),
                uv: (u, v),
                color: lattr.color,
                alpha: lattr.opacity,
            });
        }
    }

    gopts.project_sample_pass(lattr.shade_pass(unsafe { mvs.assume_init() }));
}

#[unsafe(no_mangle)]
pub extern "C" fn RiCone(height: RtFloat, radius: RtFloat, thetamax: RtFloat) {
    let mut gopts = GOPTS.lock().unwrap();
    let lattr = LATTR.read().unwrap();

    let t = lattr
        .transform_last()
        .expect("RiCone called before RiTransformBegin");
    let tdual = linear_cofactor(*t.matrix());

    let dicerate = (4 * gopts.viewport_bounds_diag(
        gopts
            .project_bounds(
                &make_corners!(radius, radius, height, 0.).map(|p| t.transform_point(&p)),
            )
            .unwrap(),
    ))
    .clamp(DICERATE_MIN, DICERATE_MAX);

    let ther = thetamax.to_radians();

    let mut mvs = Array2::uninit((dicerate + 2, dicerate + 1));
    for i in 0..mvs.dim().0 {
        let u = i as f32 / dicerate as f32;
        let (sinth, costh) = ther.algebraic_mul(u).sin_cos();
        for j in 0..mvs.dim().1 {
            let v = j as f32 / dicerate as f32;
            let zr = radius.algebraic_mul(1f32.algebraic_sub(v));

            let onorm = RtVector::new(height * costh, height * sinth, radius);
            let opos = RtPoint::new(zr * costh, zr * sinth, v * height);
            mvs[(i, j)].write(MicroVertex {
                cpos: t.transform_point(&opos),
                opos,
                cnorm: RtNormal::new_normalize(tdual * onorm),
                uv: (u, v),
                color: lattr.color,
                alpha: lattr.opacity,
            });
        }
    }

    gopts.project_sample_pass(lattr.shade_pass(unsafe { mvs.assume_init() }));
}

#[unsafe(no_mangle)]
pub extern "C" fn RiTorus(
    marad: RtFloat,
    mirad: RtFloat,
    phimin: RtFloat,
    phimax: RtFloat,
    thetamax: RtFloat,
) {
    let mut gopts = GOPTS.lock().unwrap();
    let lattr = LATTR.read().unwrap();

    let t = lattr
        .transform_last()
        .expect("RiTorus called before RiTransformBegin");
    let tdual = linear_cofactor(*t.matrix());

    let dicerate = (4 * gopts.viewport_bounds_diag(
        gopts
            .project_bounds(
                &make_corners!(marad + mirad, marad + mirad, -mirad, mirad)
                    .map(|p| t.transform_point(&p)),
            )
            .unwrap(),
    ))
    .clamp(DICERATE_MIN, DICERATE_MAX);

    let ther = thetamax.to_radians();
    let phis = phimin.to_radians();
    let phir = phimax.to_radians().algebraic_sub(phis);

    let mut mvs = Array2::uninit((dicerate + 2, dicerate + 2));
    for i in 0..mvs.dim().0 {
        let u = i as f32 / dicerate as f32;
        let (sinth, costh) = ther.algebraic_mul(u).sin_cos();
        for j in 0..mvs.dim().1 {
            let v = j as f32 / dicerate as f32;
            let (sinph, cosph) = phis.algebraic_add(phir.algebraic_mul(v)).sin_cos();
            let r = marad + mirad * cosph;

            let norm = RtVector::new(cosph * costh, cosph * sinth, sinph);
            let opos = RtPoint::new(r * costh, r * sinth, mirad * sinph);
            mvs[(i, j)].write(MicroVertex {
                cpos: t.transform_point(&opos),
                opos,
                cnorm: RtNormal::new_normalize(tdual * norm),
                uv: (u, v),
                color: lattr.color,
                alpha: lattr.opacity,
            });
        }
    }

    gopts.project_sample_pass(lattr.shade_pass(unsafe { mvs.assume_init() }));
}

/// A mesh of polygons: polygon i has nverts[i] corners, whose indices into
/// P (and N and st, when they are not null) follow one another in verts.
/// P and N hold three floats per vertex, st two. Treat every polygon as its
/// own primitive, bounded, diced, shaded and hidden before the next: a quad
/// is a bilinear patch, a triangle the same patch with its last corner
/// repeated, and a larger polygon a fan of triangles from its first vertex
/// (Section 3, "Polygon meshes"). Read the arrays with
/// std::slice::from_raw_parts.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn RiPointsPolygons(
    npolys: RtInt,
    nverts: *const RtInt,
    verts: *const RtInt,
    P: *const RtPoint,
    N: *const RtNormal,
    st: *const RtFloat,
) {
    let _ = (npolys, nverts, verts, P, N, st);
    // TODO fill in
}
