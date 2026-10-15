//! shaders.rs: the four required shaders (handout, Section 5), their
//! parameter structs (include/shaders.h, same layout), and the noise
//! function BUMPY is built on.
//!
//! CONSTANT, STANDARD, CHECKERBOARD, perlinNoise and BUMPY are stubs for you
//! to fill in, in that order. Every scene file binds the shaders by name
//! (RiSurface(CHECKERBOARD, &cp, sizeof cp)), and some call perlinNoise
//! directly, so keep the names, the signatures and the structs as they are.
//!
//! A shader is an ordinary function. The renderer fills in the inputs of the
//! RtShaderContext (ri.rs) at every grid vertex and calls the shader; a
//! surface shader writes Ci and Oi, a displacement shader moves P. The
//! context arrives as a raw pointer from C; `let c = &mut *c;` turns it into
//! a reference for the rest of the function.

#![allow(non_snake_case)] // the shader names are the handout's

use nalgebra::{Point3, SimdComplexField, UnitVector3, Vector3};

use crate::ri::{RtColor, RtFloat, RtInt, RtPoint, RtShaderContext, RtShaderFn};
use std::ffi::c_void;

/* ------------------------------------------------------------- CONSTANT */

/// The surface color, unlit: Ci = Cs and Oi = Os. The simplest shader, and
/// the first to write. FirstPrimitive binds it, so that its picture is the
/// plain RiColor whether or not your renderer runs shaders yet: once you
/// add shading, FirstPrimitive must look exactly as it did before.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn CONSTANT(c: *mut RtShaderContext) {
    let ctx = unsafe { &mut *c };
    ctx.Ci = ctx.Cs;
    ctx.Oi = ctx.Os;
}

/* ------------------------------------------------------------- STANDARD */
struct StandardLight {
    /// World-Space Position
    P: RtPoint,
    /// Color
    C: RtColor,
    /// Intensity
    I: RtFloat,
}

const DEFAULTLIGHTS: [StandardLight; 3] = [
    StandardLight {
        P: Point3::new(-6., 8., -4.),
        C: Vector3::new(1., 0.95, 0.85),
        I: 1.0,
    },
    StandardLight {
        P: Point3::new(7., 3., -2.),
        C: Vector3::new(0.6, 0.7, 1.),
        I: 0.4,
    },
    StandardLight {
        P: Point3::new(0., 5., 18.),
        C: Vector3::new(1., 0.95, 0.85),
        I: 1.0,
    },
];

struct StandardParams<'a> {
    ka: f32,
    kd: f32,
    ks: f32,
    spec_exp: usize,
    lights: &'a [StandardLight],
}

impl StandardParams<'_> {
    const fn default() -> Self {
        Self {
            ka: 0.15,
            kd: 0.8,
            ks: 0.3,
            spec_exp: 32,
            lights: &DEFAULTLIGHTS,
        }
    }
}

const DEFAULTSTANDARD: StandardParams<'static> = StandardParams::default();

/// Ambient, plus Lambertian diffuse, plus Phong specular, with the three
/// lights and the coefficients given in the handout (Section 5):
///   Ci = Cs (ka + kd sum max(N.L, 0) I C) + ks sum max(R.V, 0)^32 I C
///   Oi = Os
/// Everything is in camera space: the eye is at the origin, so the direction
/// from P toward the eye is -P. Normalize N and the light directions. Write
/// Ci as the color itself, not multiplied by Oi: the hider applies the
/// opacity when it composites (Section 7).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn STANDARD(c: *mut RtShaderContext) {
    let ctx = unsafe { &mut *c };
    let par = &DEFAULTSTANDARD;

    let mut diffuse = Vector3::zeros();
    let mut specular = Vector3::zeros();
    for light in par.lights {
        let L = UnitVector3::new_normalize(light.P - ctx.P);
        let NL = ctx.N.dot(&L);
        diffuse += NL.max(0.) * light.I * light.C;

        let R = *ctx.N * 2. * NL - *L;
        let V = UnitVector3::new_normalize(-ctx.P.coords);
        specular += R.dot(&V).max(0.).simd_powi(par.spec_exp as _) * light.I * light.C;
    }

    ctx.Ci = (par.ka * const { Vector3::new(1., 1., 1.) } + par.kd * diffuse + par.ks * specular)
        .component_mul(&ctx.Cs);
    ctx.Oi = ctx.Os;
}

/* --------------------------------------------------------- CHECKERBOARD */

/// CHECKERBOARD's parameters: sizex checks across u and sizey across v,
/// alternating between two surface shaders, light (the check at u = v = 0)
/// and dark, each run with its own parameter block. A scene writes
/// `CheckerParams cp = { 8, 4 };` (both shaders NULL, so None here) or
/// `{ 2, 2, Material, &MATTE_GREY, Material, &POLISHED_BLACK }`.
#[repr(C)]
pub struct CheckerParams {
    pub sizex: RtInt,
    pub sizey: RtInt,
    pub light: RtShaderFn,
    pub light_params: *const c_void,
    pub dark: RtShaderFn,
    pub dark_params: *const c_void,
}

/// The check (u, v) falls in is (floor(u sizex), floor(v sizey)), with u = 1
/// and v = 1 counted in the last check. Checks where i + j is even are
/// "light" and run the light sub-shader, the others the dark one. A None
/// light shader means STANDARD; a None dark shader means STANDARD with
/// Cs = black, so that only its highlights show.
///
/// You don't choose the sub-shaders: the scene does, in the CheckerParams it
/// binds, just as it passes any shader its parameters. The model scenes'
/// floors, for example, do
///     CheckerParams cp = { 2, 2, Material, &MATTE_GREY,
///                                Material, &POLISHED_BLACK };
///     RiSurface(CHECKERBOARD, &cp, sizeof cp);
/// and most scenes write { 8, 4 }, leaving both NULL (None here). Your
/// RiSurface copied that struct, so here c.params points at it. A shader is
/// a plain function, so run a check's sub-shader by calling it, with
/// c.params pointed at the sub-shader's own parameters during the call:
///     let p = &*(c.params as *const CheckerParams);
///     if let Some(light) = p.light {
///         c.params = p.light_params;
///         light(c);
///         c.params = p as *const CheckerParams as *const c_void;  (put back for whoever called CHECKERBOARD)
///     }
#[unsafe(no_mangle)]
pub unsafe extern "C" fn CHECKERBOARD(c: *mut RtShaderContext) {
    let ctx = unsafe { &mut *c };
    let par = unsafe { &*(ctx.params as *const CheckerParams) };

    if ((ctx.u * (par.sizex as f32)).floor() as isize % 2)
        == ((ctx.v * (par.sizey as f32)).floor() as isize % 2)
    {
        ctx.params = par.light_params;
        unsafe { par.light.unwrap_or(STANDARD)(ctx) }
        ctx.params = par as *const _ as *const _;
    } else {
        ctx.params = par.dark_params;
        if let Some(dark) = par.dark {
            unsafe { dark(ctx) }
        } else {
            ctx.Cs = Vector3::zeros();
            unsafe { STANDARD(ctx) }
        }

        ctx.params = par as *const _ as *const _;
    }
}

/* ---------------------------------------------------------- perlinNoise */

/// Ken Perlin's improved noise, from his paper "Improving Noise" (SIGGRAPH
/// 2002); his reference implementation, a page of Java, is at
/// https://mrl.cs.nyu.edu/~perlin/noise/. BUMPY sums octaves of it, and so
/// do some scene files' own shaders (Fbm in sceneutil.h), so until it works
/// those surfaces come out smooth and plain.
///
/// Follow his algorithm exactly: his permutation of 0..255, his fade curve
/// 6t^5 - 15t^4 + 10t^3 and his gradient function. Any other noise looks
/// just as good, but your bumps will not match reference_renders/. The
/// result is smooth and signed, about [-1, 1], and exactly 0 at every
/// integer point. Points may be negative or far from the origin, so make
/// sure the lattice index wraps into 0..255 for those too (f64 in, f64 out:
/// BUMPY's high octaves need the precision).
#[unsafe(no_mangle)]
pub extern "C" fn perlinNoise(mut x: f64, mut y: f64, mut z: f64) -> f64 {
    let X = (x.floor() as usize) & 255; // FIND UNIT CUBE THAT
    let Y = (y.floor() as usize) & 255; // CONTAINS POINT.
    let Z = (z.floor() as usize) & 255;
    x -= x.floor(); // FIND RELATIVE X,Y,Z
    y -= y.floor(); // OF POINT IN CUBE.
    z -= z.floor();
    let u = fade(x); // COMPUTE FADE CURVES
    let v = fade(y); // FOR EACH OF X,Y,Z.
    let w = fade(z);
    let A = PERLIN_PERMUTATION[X] + Y;
    let AA = PERLIN_PERMUTATION[A] + Z;
    let AB = PERLIN_PERMUTATION[A + 1] + Z; // HASH COORDINATES OF
    let B = PERLIN_PERMUTATION[X + 1] + Y;
    let BA = PERLIN_PERMUTATION[B] + Z;
    let BB = PERLIN_PERMUTATION[B + 1] + Z; // THE 8 CUBE CORNERS,

    // AND ADD BLENDED RESULTS FROM 8 CORNERS OF CUBE
    return lerp(
        w,
        lerp(
            v,
            lerp(
                u,
                grad(PERLIN_PERMUTATION[AA], x, y, z),
                grad(PERLIN_PERMUTATION[BA], x - 1., y, z),
            ),
            lerp(
                u,
                grad(PERLIN_PERMUTATION[AB], x, y - 1., z),
                grad(PERLIN_PERMUTATION[BB], x - 1., y - 1., z),
            ),
        ),
        lerp(
            v,
            lerp(
                u,
                grad(PERLIN_PERMUTATION[AA + 1], x, y, z - 1.),
                grad(PERLIN_PERMUTATION[BA + 1], x - 1., y, z - 1.),
            ),
            lerp(
                u,
                grad(PERLIN_PERMUTATION[AB + 1], x, y - 1., z - 1.),
                grad(PERLIN_PERMUTATION[BB + 1], x - 1., y - 1., z - 1.),
            ),
        ),
    );
}
fn fade(t: f64) -> f64 {
    return t * t * t * (t * (t * 6. - 15.) + 10.);
}
fn lerp(t: f64, a: f64, b: f64) -> f64 {
    return a + t * (b - a);
}
fn grad(hash: usize, x: f64, y: f64, z: f64) -> f64 {
    // CONVERT LO 4 BITS OF HASH CODE INTO 12 GRADIENT DIRECTIONS.
    let h = hash & 15;
    let u = if h < 8 { x } else { y };
    let v = if h < 4 {
        y
    } else if h == 12 || h == 14 {
        x
    } else {
        z
    };
    return if (h & 1) == 0 { u } else { -u } + if (h & 2) == 0 { v } else { -v };
}
const PERLIN_PERMUTATION: [usize; 512] = [
    151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103, 30, 69,
    142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148, 247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219,
    203, 117, 35, 11, 32, 57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175,
    74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122, 60, 211, 133, 230,
    220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54, 65, 25, 63, 161, 1, 216, 80, 73, 209, 76,
    132, 187, 208, 89, 18, 169, 200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173,
    186, 3, 64, 52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212, 207, 206,
    59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213, 119, 248, 152, 2, 44, 154, 163,
    70, 221, 153, 101, 155, 167, 43, 172, 9, 129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232,
    178, 185, 112, 104, 218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162,
    241, 81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157, 184, 84, 204,
    176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243, 141,
    128, 195, 78, 66, 215, 61, 156, 180, 151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194,
    233, 7, 225, 140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148, 247, 120, 234,
    75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32, 57, 177, 33, 88, 237, 149, 56, 87, 174,
    20, 125, 136, 171, 168, 68, 175, 74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83,
    111, 229, 122, 60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54, 65, 25,
    63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169, 200, 196, 135, 130, 116, 188,
    159, 86, 164, 100, 109, 198, 173, 186, 3, 64, 52, 217, 226, 250, 124, 123, 5, 202, 38, 147,
    118, 126, 255, 82, 85, 212, 207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170,
    213, 119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9, 129, 22, 39, 253,
    19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104, 218, 246, 97, 228, 251, 34, 242, 193,
    238, 210, 144, 12, 191, 179, 162, 241, 81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31,
    181, 199, 106, 157, 184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93,
    222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180,
];

pub fn perlin_noise(x: nalgebra::Point3<f64>) -> f64 {
    perlinNoise(x.x, x.y, x.z)
}

/* ---------------------------------------------------------------- BUMPY */

/// BUMPY's parameters: see BUMPY below.
#[repr(C)]
pub struct BumpParams {
    pub amplitude: RtFloat,
    pub min_freq_exp: RtInt,
    pub max_freq_exp: RtInt,
}
/// Displace P along the unit normal N by
///   h = amplitude * sum(w_e * perlinNoise(2^e * Po)) / sum(w_e)
/// over e = min_freq_exp ..= max_freq_exp, with w_e = 2^(min_freq_exp - e):
/// each octave counts half as much as the one below it, and dividing by the
/// total weight keeps |h| <= amplitude whatever the octaves. Evaluate the
/// noise at Po (object space), so the bumps stick to the object as it
/// moves. The parameters are a BumpParams at c.params.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn BUMPY(c: *mut RtShaderContext) {
    let ctx = unsafe { &mut *c };
    let par = unsafe { &*(ctx.params as *const BumpParams) };

    let n = par.max_freq_exp - par.min_freq_exp;
    let scale = 2.0 * (1.0 - (2 << -1 - n) as f32);

    let mut h = 0.0;
    let mut ampl = 1.0;
    let mut freq = (2 << par.min_freq_exp as usize) as f32 * ctx.Po;

    for _ in 0..=n {
        h += perlin_noise(freq.cast()) as f32 * ampl;
        ampl *= 0.5;
        freq *= 2.0;
    }

    ctx.P += ctx.N.scale(h * par.amplitude / scale);
}
