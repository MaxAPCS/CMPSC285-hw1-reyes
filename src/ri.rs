//! ri.rs: the HW1 RenderMan-like interface (include/Ri.h) in Rust, and the
//! renderer behind it.
//!
//! The scene files are C++ and compile against Ri.h; this file declares the
//! same types with the same memory layout (#[repr(C)]) and exports the same
//! functions with C linkage (#[no_mangle] extern "C"), so the unchanged
//! scene files link against your renderer. Keep every signature as it is.
//!
//! This is a stub. Every function the scenes call is here, does nothing
//! useful yet, and RiWorldEnd writes a black image of the right size and
//! name, so the whole kit builds and runs before you have written a line.
//! Replace the stubs with your REYES renderer, following the Suggested
//! Workflow: transformations and the camera first, then one primitive
//! (FirstPrimitive_sphere is the scene for it), then dicing, sampling, the
//! hider, the other primitives, shading, textures, displacement, opacity,
//! and depth output. The comment on each function says what the finished
//! version must do and where the handout describes it. Split the code into
//! more modules as it grows.
//!
//! The two functions at the end, write_image and write_depth, are complete:
//! they write the file RiDisplay named, in the format its extension says,
//! from a buffer you fill in. Nothing else here is.
//!
//! The shaders live in shaders.rs. Everything the scene files include is in
//! ../include/ and must not change: Ri.h, shaders.h and scenes.h are the
//! interface; obj.h, shadowmap.h, sceneutil.h, anim.h and models.h are
//! helpers for the scene files that never call back into the renderer
//! except through that interface.

#![allow(non_snake_case)] // the Ri names and the context's fields are RenderMan's

use std::collections::BinaryHeap;
use std::ffi::{CStr, c_char, c_void};
use std::io::Write;
use std::mem::MaybeUninit;
use std::path::Path;
use std::ptr;
use std::sync::{Arc, Mutex};
use std::{debug_assert_matches, f32};

use image::{DynamicImage, GenericImageView, Pixel};
use nalgebra::{Matrix4, Orthographic3, Perspective3, Projective3, UnitVector3, Vector3, matrix};
use ndarray::{Array1, Array2, Array3, Axis, Zip, s};

use crate::shaders;

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

/// What a shader sees at one grid vertex (Ri.h's RtShaderContext, same
/// layout). The renderer fills in the inputs; a displacement shader moves
/// P, a surface shader writes Ci and Oi.
#[repr(C)]
pub struct RtShaderContext {
    /* inputs, filled by the renderer at each grid vertex */
    pub P: RtPoint,  /* position of grid vertex in camera/eye space */
    pub Po: RtPoint, /* same point in object space */
    pub N: RtNormal, /* unit normal, recomputed after displacement */
    pub u: RtFloat,  /* surface parameters in [0,1], or a mesh's st */
    pub v: RtFloat,
    pub Cs: RtColor, /* current RiColor and RiOpacity */
    pub Os: RtColor,
    pub params: *const c_void, /* this shader's parameter block, or null */
    /* outputs */
    pub Ci: RtColor, /* written by surface shaders */
    pub Oi: RtColor,
    /* displacement shaders modify P in place */
}

/// A shader: a C function pointer, or None where a scene passes NULL
/// (RiSurface(NULL, NULL, 0) means STANDARD).
pub type RtShaderFn = Option<unsafe extern "C" fn(*mut RtShaderContext)>;

/// A C string argument as a Rust &str ("" for a null pointer).
#[inline]
unsafe fn read_token(s: RtToken) -> &'static str {
    if s.is_null() {
        ""
    } else {
        unsafe { CStr::from_ptr(s) }.to_str().unwrap_or("")
    }
}

/* --------------------------------------------------------- renderer state */

/// Running state, changes model-to-model
#[derive(Clone)]
struct ModelState {
    transform: Vec<RtMatrix>,
    color: RtColor,
    opacity: RtColor,
    surface: RtShaderFn,
    surface_params: Option<Arc<[u8]>>,
    displace: RtShaderFn,
    displace_params: Option<Arc<[u8]>>,

    projection: Option<Projective3<RtFloat>>,
    textures: [Option<DynamicImage>; 8],
}

const WHITE: RtColor = RtColor::new(1., 1., 1.);
impl ModelState {
    const fn default() -> Self {
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
}
impl Default for ModelState {
    fn default() -> Self {
        Self::default()
    }
}

/// What the scene has asked for so far. The finished renderer keeps much
/// more here: the current transformation, its stack, the color, opacity and
/// bound shaders, the projection, the sample buffer ... (handout, Section 2).
struct RenderState {
    xres: usize,
    yres: usize,
    xsamples: usize,
    ysamples: usize,
    display: String,
    depth_display: bool, /* RiDisplay(name, "zfile", "z") */
    depth_buffer: Option<Array2<f32>>,
    sample_buffer: Option<Array2<BinaryHeap<Sample>>>,
}
impl RenderState {
    const fn default() -> Self {
        Self {
            xres: 640,
            yres: 480,
            xsamples: 2,
            ysamples: 2,
            display: String::new(),
            depth_display: false,
            depth_buffer: None,
            sample_buffer: None,
        }
    }
}
impl Default for RenderState {
    fn default() -> Self {
        Self::default()
    }
}

struct MicroPolygon {
    pub P: RtPoint,  /* point in world space */
    pub Po: RtPoint, /* object space */
    pub N: RtVector, /* area-scaled normal, recomputed after displacement */
    pub u: RtFloat,  /* surface parameters in [0,1], or a mesh's st */
    pub v: RtFloat,
    pub Cs: RtColor, /* current RiColor and RiOpacity */
    pub Os: RtColor,
}
impl MicroPolygon {
    fn surface(&mut self, shader: RtShaderFn, params: Option<Arc<[u8]>>) {
        let params = if let Some(params) = params {
            params.as_ptr()
        } else {
            ptr::null()
        } as _;
        let mut ctx = RtShaderContext {
            P: self.P,
            Po: self.Po,
            N: RtNormal::new_normalize(self.N),
            u: self.u,
            v: self.v,
            Cs: self.Cs,
            Os: self.Os,
            params,
            Ci: RtColor::zeros(),
            Oi: RtColor::zeros(),
        };
        unsafe { shader.unwrap_or(shaders::STANDARD)(ptr::from_mut(&mut ctx)) }
        self.Cs = ctx.Ci;
        self.Os = ctx.Oi;
    }
    fn displace(&mut self, shader: RtShaderFn, params: Option<Arc<[u8]>>) {
        let Some(shader) = shader else { return };
        let params = if let Some(params) = params {
            params.as_ptr()
        } else {
            ptr::null()
        } as _;
        let mut ctx = RtShaderContext {
            P: self.P,
            Po: self.Po,
            N: RtNormal::new_normalize(self.N),
            u: self.u,
            v: self.v,
            Cs: self.Cs,
            Os: self.Os,
            params,
            Ci: RtColor::zeros(),
            Oi: RtColor::zeros(),
        };
        unsafe { shader(ptr::from_mut(&mut ctx)) }
        self.P = ctx.P;
        // must recompute normals later
    }
}

#[derive(Clone)] // exclusively so populating the Array2 works
struct Sample {
    z: RtFloat,
    Cs: RtColor,
    Os: RtColor,
}
impl PartialEq for Sample {
    fn eq(&self, other: &Self) -> bool {
        self.z == other.z
    }
}
impl Eq for Sample {}
impl PartialOrd for Sample {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.z.partial_cmp(&other.z)
    }
}
impl Ord for Sample {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.z.total_cmp(&other.z)
    }
}

const IDENTITY: RtMatrix =
    RtMatrix::from_matrix_unchecked(matrix! [1.,0.,0.,0.;0.,1.,0.,0.;0.,0.,1.,0.;0.,0.,0.,1.]);

static MSTATE: Mutex<ModelState> = Mutex::new(ModelState::default());
static RSTATE: Mutex<RenderState> = Mutex::new(RenderState::default());
static FRAME_MSTATE: Mutex<Option<ModelState>> = Mutex::new(None);

/* ---------------------------------------------------------- graphics state */

/// RiBegin starts a rendering session and resets the graphics state to
/// RenderMan's defaults: color and opacity (1, 1, 1), surface shader
/// STANDARD, no displacement, orthographic projection, 640 x 480 pixels,
/// 2 x 2 samples. Scenes rely on these (FirstPrimitive never calls
/// RiOpacity). RiEnd ends the session (free textures and buffers here).
/// The name is always RI_NULL.
#[unsafe(no_mangle)]
pub extern "C" fn RiBegin(_name: RtToken) {
    let mut rstate = RSTATE.lock().unwrap();
    *rstate = RenderState::default();
    let mut mstate = MSTATE.lock().unwrap();
    *mstate = ModelState::default();
}

#[unsafe(no_mangle)]
pub extern "C" fn RiEnd() {}

/// The image size in pixels. The screen window is [-a, a] x [-1, 1] for
/// a = xres / yres >= 1 and [-1, 1] x [-1/a, 1/a] for a < 1, so the field of
/// view spans the shorter side (Section 2).
#[unsafe(no_mangle)]
pub extern "C" fn RiFormat(xres: RtInt, yres: RtInt) {
    let mut rstate = RSTATE.lock().unwrap();
    rstate.xres = xres as usize;
    rstate.yres = yres as usize;
}

/// xsamples x ysamples jittered samples per pixel (Section 4).
#[unsafe(no_mangle)]
pub extern "C" fn RiPixelSamples(xsamples: RtInt, ysamples: RtInt) {
    let mut rstate = RSTATE.lock().unwrap();
    rstate.xsamples = xsamples.max(1) as usize;
    rstate.ysamples = ysamples.max(1) as usize;
}

/// The output file, written into renders/ by RiWorldEnd. RiDisplay(name, "",
/// "") names the picture; RiDisplay(name, "zfile", "z") asks for the depth
/// image instead (Section 4, "Depth output").
#[unsafe(no_mangle)]
pub unsafe extern "C" fn RiDisplay(name: *const c_char, type_: RtToken, mode: RtToken) {
    let mut rstate = RSTATE.lock().unwrap();
    unsafe {
        rstate.display = read_token(name).to_string();
        rstate.depth_display = read_token(type_) == "zfile" || read_token(mode) == "z";
    }
}

/// RiFrameBegin/End bracket one frame of an animation: save the graphics
/// state at Begin and restore it at End, so that each frame starts afresh.
#[unsafe(no_mangle)]
pub extern "C" fn RiFrameBegin(_frame: RtInt) {
    let mstate = MSTATE.lock().unwrap();
    let mut fmstate = FRAME_MSTATE.lock().unwrap();

    *fmstate = Some(mstate.clone());
}

#[unsafe(no_mangle)]
pub extern "C" fn RiFrameEnd() {
    let mut mstate = MSTATE.lock().unwrap();
    let fmstate = FRAME_MSTATE
        .lock()
        .unwrap()
        .take()
        .expect("RiFrameEnd called before RiFrameBegin");
    *mstate = fmstate;
}

/// RiWorldBegin: the transformation built so far is the camera transform
/// (world to camera); from here on transformations place objects. Set up
/// the sample buffer.
#[unsafe(no_mangle)]
pub extern "C" fn RiWorldBegin() {
    let mut mstate = MSTATE.lock().unwrap();
    let tf = mstate
        .transform
        .pop()
        .expect("RiWorldBegin called before RiProjection");
    assert!(mstate.projection.is_none());
    mstate.projection = Some(tf);
    mstate.transform = vec![IDENTITY];

    let mut rstate = RSTATE.lock().unwrap();
    rstate.sample_buffer = Some(Array2::from_elem(
        (rstate.yres * rstate.ysamples, rstate.xres * rstate.xsamples),
        BinaryHeap::new(),
    ));
    rstate.depth_buffer = Some(Array2::from_elem((rstate.yres, rstate.xres), 0.));
}

/// RiWorldEnd: all primitives have been drawn, so filter the samples into pixels and write the image (Sections 2 and 4).
#[unsafe(no_mangle)]
pub extern "C" fn RiWorldEnd() {
    let mut rstate = RSTATE.lock().unwrap();
    if rstate.depth_display {
        let dbuf = rstate
            .depth_buffer
            .take()
            .expect("RiWorldEnd called before RiWorldBegin");
        write_depth(&dbuf, &rstate.display);
    } else {
        let mut fbuf = Array3::<u8>::uninit((rstate.yres, rstate.xres, 4));
        {
            let mut sbuf = rstate
                .sample_buffer
                .take()
                .expect("RiWorldEnd caleld before RiWorldBegin");
            sbuf.par_map_inplace(|s| {
                if s.is_empty() {
                    return;
                }

                let mut opacity = RtColor::zeros();
                let mut color = RtColor::zeros();
                while let Some(sample) = s.pop() {
                    opacity += sample.Os;
                    color += sample.Cs.component_mul(&sample.Os);
                    if opacity == WHITE {
                        s.clear();
                        break;
                    }
                }
                s.push(Sample {
                    z: 0.,
                    Cs: color,
                    Os: opacity,
                });
            });
            Zip::from(sbuf.exact_chunks_mut((rstate.ysamples, rstate.xsamples)))
                .and(fbuf.lanes_mut(Axis(2)))
                .par_for_each(|samples, mut lane| {
                    let mut color = Array1::from_elem(4, 0.);
                    for s in samples {
                        if let Some(s) = s.pop() {
                            color[0] += s.Cs.x;
                            color[1] += s.Cs.y;
                            color[2] += s.Cs.z;
                            color[3] += s.Os.sum();
                        }
                    }
                    color /= (rstate.ysamples * rstate.xsamples) as RtFloat;
                    color[3] /= 3.; // average of 3 opacity colors
                    lane.assign((*color.mapv(to8).mapv(MaybeUninit::new)).into());
                });
        } // drop sbuf
        write_image(unsafe { fbuf.assume_init() }, &rstate.display).unwrap();
    }
}

/// RiProjection(RI_PERSPECTIVE, "fov", &fov), fov in degrees over the
/// shorter side of the image, or RiProjection(RI_ORTHOGRAPHIC, NULL, NULL):
/// the C++ scenes omit the last two arguments, which default to NULL. It
/// also resets the current transformation to the identity: the camera
/// transform is built after it (Section 2). value points at an RtFloat.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn RiProjection(name: RtToken, _token: RtToken, value: RtPointer) {
    let name = unsafe { read_token(name) };

    let mut mstate = MSTATE.lock().unwrap();
    let rstate = RSTATE.lock().unwrap();
    let (xr, yr) = (rstate.xres as f32, rstate.yres as f32);
    mstate.transform = vec![match name {
        "perspective" => Perspective3::new(
            yr / xr,
            unsafe { *(value as *const _ as *const _) },
            0.1,
            100.,
        )
        .to_projective(),
        "orthographic" => Orthographic3::new(-xr, xr, -yr, yr, 0.1, 100.).to_projective(),
        _ => panic!("RiProjection called with invalid name"),
    }];
    mstate.projection = None;
}

/* ----------------------------------------------------------- transformations */

/// The current transformation matrix (CTM) and its stack. Each call
/// post-multiplies the CTM, so the transformation called last is applied
/// to points first; RiTransformBegin pushes a copy and RiTransformEnd pops
/// it. Scenes nest these several deep. RiConcatTransform takes an RtMatrix
/// in RenderMan's row-vector form (p' = p M, translation in the bottom
/// row); in C an array argument decays to a pointer, so it arrives as a
/// pointer to four rows of four floats. Angles are degrees (Section 2).

/// Transform push
#[unsafe(no_mangle)]
pub extern "C" fn RiTransformBegin() {
    let tstk = &mut MSTATE.lock().unwrap().transform;
    let t = tstk.last().cloned().unwrap_or(IDENTITY);
    tstk.push(t);
}

/// Transform pop
#[unsafe(no_mangle)]
pub extern "C" fn RiTransformEnd() {
    let tstk = &mut MSTATE.lock().unwrap().transform;
    tstk.pop()
        .expect("RiTransformEnd called before RiWorldBegin or RtProjection");
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn RiConcatTransform(transform: *const [RtFloat; 4]) {
    let tstk = &mut MSTATE.lock().unwrap().transform;
    let transform: RtMatrix = bytemuck::cast(unsafe { *transform });
    // assert!(transform.is_invertible());
    // let transform = RtMatrix::from_matrix_unchecked(transform);
    *tstk
        .last_mut()
        .expect("RiConcatTransform called before RiWorldBegin or RtProjection") *= transform;
}

#[unsafe(no_mangle)]
pub extern "C" fn RiTranslate(dx: RtFloat, dy: RtFloat, dz: RtFloat) {
    let tstk = &mut MSTATE.lock().unwrap().transform;
    tstk.last_mut()
        .expect("RiTranslate called before RiWorldBegin or RtProjection")
        .matrix_mut_unchecked()
        .append_translation_mut(&RtVector::new(dx, dy, dz));
}

#[unsafe(no_mangle)]
pub extern "C" fn RiRotate(angle: RtFloat, dx: RtFloat, dy: RtFloat, dz: RtFloat) {
    let tstk = &mut MSTATE.lock().unwrap().transform;
    *tstk
        .last_mut()
        .expect("RiTranslate called before RiTransformBegin")
        .matrix_mut_unchecked() *=
        Matrix4::from_axis_angle(&RtNormal::new_unchecked(RtVector::new(dx, dy, dz)), angle);
}

#[unsafe(no_mangle)]
pub extern "C" fn RiScale(sx: RtFloat, sy: RtFloat, sz: RtFloat) {
    let tstk = &mut MSTATE.lock().unwrap().transform;
    tstk.last_mut()
        .expect("RiTranslate called before RiWorldBegin or RtProjection")
        .matrix_mut_unchecked()
        .append_nonuniform_scaling_mut(&RtVector::new(sx, sy, sz));
}

/* ---------------------------------------------------------------- primitives */

/// Each primitive goes through the REYES pipeline as soon as it is called:
/// bound it, choose a dice rate from the bound's size on screen, dice it
/// into a grid of micropolygons, shade the grid, and hand the micropolygons
/// to the hider (Section 3). The quadrics use the RenderMan parameterizations
/// (RiSpec section 5.4); thetamax, phimin and phimax are degrees.
///
/// Shade the grid in two passes (Section 5): the displacement shader at
/// every vertex, then N recomputed from the displaced grid, then the surface
/// shader at every vertex. Each call fills one vertex's RtShaderContext,
/// runs the shader and keeps what it wrote. A shader sees only its own
/// vertex, so the order you visit them in does not matter; only the new N
/// needs the neighbors, which is why it waits for the whole first pass.
#[unsafe(no_mangle)]
pub extern "C" fn RiSphere(radius: RtFloat, zmin: RtFloat, zmax: RtFloat, thetamax: RtFloat) {
    let mstate = MSTATE.lock().unwrap();
    let t = mstate
        .transform
        .last()
        .expect("RiSphere called before RiTransformBegin");
    let p = mstate
        .projection
        .expect("RiSphere called before RiWorldBegin");

    let mut rstate = RSTATE.lock().unwrap();
    let sbuf = rstate
        .sample_buffer
        .as_mut()
        .expect("RiSphere called before RiWorldBegin");
    let (sbrows, sbcols, ..) = sbuf.dim();

    let ther = thetamax.to_radians();
    let phis = zmin.asin();
    let phir = zmax.asin() - phis;

    let dicerate = 100;
    // theta range / 2pi * phi range / pi * 4pir^2 / dicerate^2
    let msize = (radius * radius)
        .algebraic_mul(phir)
        .algebraic_mul(thetamax)
        .algebraic_mul(const { 4. / 360. })
        .algebraic_div((dicerate * dicerate) as _);

    for i in 0..dicerate {
        let u = i as f32 / dicerate as f32;
        let theta = ther.algebraic_mul(u as _);
        let (sinth, costh) = theta.sin_cos();

        for j in 0..dicerate {
            let v = j as f32 / dicerate as f32;
            let phi = phis.algebraic_add(phir.algebraic_mul(v as _));
            let (sinph, cosph) = phi.sin_cos();

            let norm =
                UnitVector3::new_unchecked(Vector3::new(sinth * cosph, sinth * sinph, costh));
            let rpos = RtPoint::from(norm.scale(radius)); // object-space
            let mut mp = MicroPolygon {
                P: t.transform_point(&rpos),
                Po: rpos,
                N: t.transform_vector(&norm).normalize(),
                u,
                v,
                Cs: mstate.color,
                Os: mstate.opacity,
            };
            mp.displace(mstate.displace, mstate.displace_params.clone()); // TODO recompute norms after
            mp.surface(mstate.surface, mstate.surface_params.clone());

            let Pss: RtPoint = p.transform_point(&mp.P);
            sbuf[(
                ((Pss.x + 1.) / 2. * sbrows as f32) as usize,
                ((Pss.y + 1.) / 2. * sbcols as f32) as usize,
            )]
                .push(Sample {
                    z: Pss.z,
                    Cs: mp.Cs,
                    Os: mp.Os,
                });
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn RiCylinder(radius: RtFloat, zmin: RtFloat, zmax: RtFloat, thetamax: RtFloat) {
    let _ = (radius, zmin, zmax, thetamax);
}

#[unsafe(no_mangle)]
pub extern "C" fn RiCone(height: RtFloat, radius: RtFloat, thetamax: RtFloat) {
    let _ = (height, radius, thetamax);
}

#[unsafe(no_mangle)]
pub extern "C" fn RiTorus(
    majorradius: RtFloat,
    minorradius: RtFloat,
    phimin: RtFloat,
    phimax: RtFloat,
    thetamax: RtFloat,
) {
    let _ = (majorradius, minorradius, phimin, phimax, thetamax);
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
}

/* ------------------------------------------------------------------ shading */

/// Attributes: they stay set until changed (RiTransformEnd does not restore
/// them). The color and opacity become Cs and Os in the shader context.
/// An RtColor argument arrives as a pointer to its three floats.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn RiColor(color: *const RtFloat) {
    let mut s = MSTATE.lock().unwrap();
    let color: &RtColor = unsafe { &*(color as *const _) };
    s.color = *color;
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn RiOpacity(opacity: *const RtFloat) {
    let mut s = MSTATE.lock().unwrap();
    let opacity: &RtColor = unsafe { &*(opacity as *const _) };
    s.opacity = *opacity;
}

/// Bind a shader to the primitives that follow. Copy nbytes of params now
/// (std::slice::from_raw_parts(params as *const u8, nbytes).to_vec()): the
/// scene's struct may be gone by the time the shader runs. A None shader
/// means STANDARD for RiSurface and no displacement for RiDisplacement
/// (Section 5).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn RiSurface(shader: RtShaderFn, params: *const c_void, nbytes: usize) {
    let mut s = MSTATE.lock().unwrap();
    s.surface = shader;
    s.surface_params = if !params.is_null() && nbytes > 0 {
        // SAFETY: params null check, u8 is always valid
        let params = unsafe { std::slice::from_raw_parts(params as *const u8, nbytes) };
        Some(params.into())
    } else {
        None
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn RiDisplacement(shader: RtShaderFn, params: *const c_void, nbytes: usize) {
    let mut s = MSTATE.lock().unwrap();
    s.displace = shader;
    s.displace_params = if !params.is_null() && nbytes > 0 {
        // SAFETY: params null check, u8 is always valid
        let params = unsafe { std::slice::from_raw_parts(params as *const u8, nbytes) };
        Some(params.into())
    } else {
        None
    }
}

/// Load an image into one of 8 texture slots with the image crate
/// (image::open; look for the file as given, then under textures/)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn RiMakeTexture(file: *const c_char, slot: RtInt) {
    let path = unsafe { read_token(file) };
    let image = image::open(path).expect("RiMakeTexture called with invalid file");

    let mut mstate = MSTATE.lock().unwrap();
    mstate.textures[slot as usize] = Some(image);
}

/// look it up bilinearly with repeat wrap: s runs left to right, t from the
/// TOP row (t = 0) to the bottom (Section 6). An empty slot reads black.
/// result points at three floats.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn RiTexture(slot: RtInt, s: RtFloat, t: RtFloat, result: *mut RtFloat) {
    let result: &mut RtVector = unsafe { &mut *(result as *mut _) };

    let mstate = MSTATE.lock().unwrap();
    let image = mstate.textures[slot as usize]
        .as_ref()
        .expect("RiTexture called with invalid slot");

    let (w, h) = image.dimensions();
    let x = (s - s.floor()) * w as RtFloat;
    let y = (t - t.floor()) * h as RtFloat;

    result.fill(0.);
    for sx in [x.floor(), x.ceil()] {
        for sy in [y.floor(), y.ceil()] {
            *result += (1.0 - (x - sx).abs())
                * (1.0 - (y - sy).abs())
                * RtVector::from_row_slice(
                    &image
                        .get_pixel(sx as _, sy as _)
                        .to_rgb()
                        .0
                        .map(|x| x as RtFloat / u8::MAX as RtFloat),
                );
        }
    }
}

/* ------------------------------------------------------------ output files */

/// Lowercase extension of a file name, without the dot ("" if none).
fn extension(name: &str) -> String {
    Path::new(name)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default()
}

/// A value clamped to [0, 1], as an 8-bit integer.
#[inline]
fn to8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// Write the picture to renders/<RiDisplay name>, in the format its
/// extension names. Complete: call it from RiWorldEnd with your filtered
/// pixels, xres * yres * 4 floats, top row first: each pixel's color as
/// compositing leaves it, premultiplied by alpha (that is, over the black
/// background), then its alpha, the mean of its three opacity channels.
/// Don't clamp or convert them; this does, per format:
///   .png, .tga  keep alpha. Both store straight color, so each pixel's
///               color is divided by its alpha: a pixel half covered by a
///               white surface is stored as white with alpha one half, and
///               whatever shows the file multiplies the alpha back in.
///   .jpg, .bmp  are written without alpha: JPEG has none, and most
///               programs ignore a BMP's. They get the picture over the
///               black background, which is the premultiplied color itself.
/// Values are clamped to [0, 1] and rounded to 8 bits.
pub fn write_image(rgba: Array3<u8>, name: &str) -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("renders/{}", name);
    let ext = extension(&name);
    let keep_alpha = match ext.as_str() {
        "png" | "tga" => true,
        "jpg" | "jpeg" | "bmp" => false,
        _ => {
            return Err(format!("cannot write {}: use .png, .bmp, .tga or .jpg", path).into());
        }
    };

    let (rgbar, rgbac, depth) = rgba.dim();
    debug_assert_eq!(depth, 4);
    if keep_alpha {
        let (raw, off) = rgba.into_raw_vec_and_offset();
        debug_assert_matches!(off, Some(0) | None);

        let image = image::RgbaImage::from_raw(rgbac as _, rgbar as _, raw)
            .ok_or("write_image called with invalid buffer")?;
        Ok(image.save_with_format(
            &path,
            if ext == "png" {
                image::ImageFormat::Png
            } else {
                image::ImageFormat::Tga
            },
        )?)
    } else {
        let rgba = rgba.mapv(|x| x as u16);
        let rgb = rgba.slice(s![.., .., 0..3]); // (H, W, 3) view  
        let alpha = rgba.slice(s![.., .., 3..4]); // (H, W, 1) view  
        let rgba = ((&rgb * &alpha) / 255).mapv(|x| x as u8).to_owned(); // broadcast → (H, W, 3)
        debug_assert_eq!(rgba.dim().2, 3);

        let (raw, off) = rgba.into_raw_vec_and_offset();
        debug_assert_matches!(off, Some(0) | None);

        let image = image::RgbImage::from_raw(rgbac as _, rgbar as _, raw)
            .ok_or("write_image called with invalid buffer")?;
        Ok(if ext == "bmp" {
            image.save_with_format(&path, image::ImageFormat::Bmp)?
        } else {
            std::fs::File::create(&path)
                .map_err(image::ImageError::IoError)
                .and_then(|file| {
                    let mut out = std::io::BufWriter::new(file);
                    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 95)
                        .encode_image(&image)
                })?
        })
    }
}

/// Write the depth image to renders/<RiDisplay name> in the handout's
/// format: "Rz01", int32 width and height, then one float per pixel, top
/// row first, all little-endian (the camera-space z of the nearest surface
/// at least half opaque, +infinity where there is none). Complete: call it
/// from RiWorldEnd when RiDisplay asked for depth. The scene that reads the
/// file back knows the camera it was rendered with.
pub fn write_depth(depth: &Array2<f32>, path: &str) {
    let result = std::fs::File::create(&path).and_then(|file| {
        let mut out = std::io::BufWriter::new(file);
        out.write_all(b"Rz01")?;
        out.write_all(&depth.dim().0.to_le_bytes())?;
        out.write_all(&depth.dim().1.to_le_bytes())?;
        for d in depth {
            out.write_all(&d.to_le_bytes())?;
        }
        out.flush()
    });
    match result {
        Ok(()) => println!(
            "wrote {} ({} x {} depth)",
            path,
            depth.dim().0,
            depth.dim().1
        ),
        Err(e) => eprintln!(
            "could not write {} ({}; run reyes from the kit root)",
            path, e
        ),
    }
}
