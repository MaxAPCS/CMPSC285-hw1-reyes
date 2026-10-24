#![allow(non_snake_case)]
mod bind {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

use std::ptr::null;

use crate::{Scene, ri_api::*, shaders::*};

pub static SCENE: Scene = Scene {
    name: "CustomScene",
    tier: "cinematic",
    function: Function,
};

unsafe extern "C" fn Function() {
    function();
}

const DOWNSCALE: i32 = 1;

#[inline]
fn function() {
    RiBegin(null());
    {
        ri_maketexture("stars.png", 0);
        ri_maketexture("mars.png", 1);

        ri_display("CustomScene.png".to_string(), false);
        RiFormat(1920 / DOWNSCALE, 1080 / DOWNSCALE);
        RiPixelSamples(4 / DOWNSCALE, 4 / DOWNSCALE);
        ri_projection("perspective", Some(180.));
        RiFrameBegin(0);
        {
            RiWorldBegin();
            {
                let shaderparams = StandardParams {
                    ka: 1.,
                    kd: 0.0,
                    ks: 0.0,
                    spec_exp: 0,
                    lights: &[],
                };

                // Stars
                RiTransformBegin();
                {
                    RiOpacity(&WHITE);
                    RiColor(&WHITE);
                    let shaderparams = TexturedStandardParams::new(0, &shaderparams);
                    RiSurface(
                        Some(TEXSTANDARD),
                        &shaderparams as *const _ as *const _,
                        std::mem::size_of::<TexturedStandardParams>(),
                    );
                    RiTranslate(0., 0., 750.);
                    ri_quad(
                        [
                            RtPoint::new(-2700., -1500., 0.),
                            RtPoint::new(-2700., 1500., 0.),
                            RtPoint::new(2700., 1500., 0.),
                            RtPoint::new(2700., -1500., 0.),
                        ],
                        None,
                        None,
                    );
                }
                RiTransformEnd();

                // Orbital Plane
                RiTransformBegin();
                {
                    RiTranslate(400., 20., 256.);
                    RiRotate(-90., 1., 0., 0.);
                    RiRotate(170., 0., 0., 1.);

                    RiTransformBegin();
                    {
                        RiTranslate(600., 0., -50.);
                        RiSurface(
                            Some(TEXSTANDARD),
                            &TexturedStandardParams::new(
                                1,
                                &StandardParams {
                                    ka: 0.7,
                                    kd: 0.9,
                                    ks: 0.3,
                                    spec_exp: 4,
                                    lights: &[StandardLight {
                                        P: RtPoint::new(400., 20., 256.),
                                        C: RtColor::new(1.0, 0.8, 0.5),
                                        I: 0.4,
                                    }],
                                },
                            ) as *const _ as *const _,
                            std::mem::size_of::<TexturedStandardParams>(),
                        );
                        RiDisplacement(
                            Some(BUMPY),
                            &BumpParams {
                                amplitude: 0.6,
                                min_freq_exp: 0,
                                max_freq_exp: 4,
                            } as *const _ as *const _,
                            std::mem::size_of::<BumpParams>(),
                        );
                        RiSphere(80., -80., 80., 360.);
                    }
                    RiTransformEnd();
                    RiTransformBegin();
                    {
                        RiColor(&RtColor::new(0.3, 0.3, 0.3));
                        RiSurface(
                            Some(SCATTER),
                            &ScatterParams { threshold: 0.3 } as *const _ as *const _,
                            std::mem::size_of::<ScatterParams>(),
                        );
                        RiRotate(6., 0., 1., 0.);
                        RiTorus(650., 80., 0., 180., 360.);
                    }
                    RiTransformEnd();

                    RiColor(&RtColor::new(1.0, 0.8, 0.5));
                    RiSurface(
                        Some(STANDARD),
                        &shaderparams as *const _ as *const _,
                        std::mem::size_of::<StandardParams>(),
                    );
                    RiSphere(48., -48., 48., 360.);
                }
                RiTransformEnd();

                let shaderparams = StandardParams {
                    ka: 0.2,
                    kd: 1.0,
                    ks: 0.6,
                    spec_exp: 4,
                    lights: &[StandardLight {
                        P: RtPoint::new(180., 16., 128.),
                        C: RtColor::new(0.9, 1., 0.4),
                        I: 1.,
                    }],
                };
                RiSurface(
                    Some(STANDARD),
                    &shaderparams as *const _ as *const _,
                    std::mem::size_of::<StandardParams>(),
                );
                RiDisplacement(None, null(), 0);

                // Windshield
                RiTransformBegin();
                {
                    RiColor(&RtColor::from_element(1.));
                    RiOpacity(&RtColor::new(0.4, 0.4, 0.7));
                    RiTranslate(0., 0., -5.);
                    RiRotate(45., 1., 0., 0.);
                    RiSphere(8., 0., 8., 180.);
                }
                RiTransformEnd();

                // Console
                RiTransformBegin();
                {
                    RiColor(&RtColor::new(0.3, 0.3, 0.3));
                    RiOpacity(&RtColor::from_element(1.));
                    RiTranslate(0., 1.8, 0.5);
                    RiRotate(-90., 0., 1., 0.);
                    RiCylinder(0.7, -3.4, 3.4, 35.);
                }
                RiTransformEnd();

                // Button
                RiTransformBegin();
                {
                    RiColor(&RtColor::new(1.0, 0.0, 0.0));
                    RiTranslate(0., 2.0, 1.1);
                    RiRotate(90., 1., 0., 0.);
                    RiScale(1., 0.7, 0.4);
                    RiSphere(0.2, 0., 0.2, 360.);
                }
                RiTransformEnd();

                // Nosecone
                RiTransformBegin();
                {
                    RiColor(&RtColor::new(0.9, 0.3, 0.3));
                    RiOpacity(&RtColor::from_element(1.));
                    RiTranslate(0., 2.5, 0.25);
                    RiRotate(-90., 1., 0., 0.);
                    RiCone(0.01, 4., 360.);
                }
                RiTransformEnd();

                // Buzz Lightyear??
                // RiTransformBegin();
                // unsafe {
                //     RiTranslate(-3., -1., 9.);
                //     RiRotate(260., 1., 1., 2.);
                //     let buzz = bind::obj_load("models/buzz/buzz.obj".as_ptr().cast());

                //     for k in 0..(*buzz).ngroups as usize {
                //         let g = &*((*buzz).groups.add(k));
                //         if CStr::from_bytes_with_nul(&*(&g.material as *const _ as *const [u8; 64]))
                //             .unwrap()
                //             .to_str()
                //             .unwrap()
                //             == "dome"
                //         {
                //             RiColor(&RtColor::new(0.85, 0.92, 1.0));
                //             RiOpacity(&RtColor::new(0.2, 0.2, 0.2));
                //             RiSurface(None, null(), 0);
                //             bind::BuzzLenses(g);
                //             bind::BuzzHelmet();
                //         } else {
                //             /* body and face */
                //             RiColor(&WHITE);
                //             RiOpacity(&WHITE);
                //             // RiSurface(TexturedStandard, &tp, sizeof tp);
                //             RiPointsPolygons(
                //                 g.npolys,
                //                 g.nverts,
                //                 g.verts,
                //                 g.P as *const _ as *const _,
                //                 g.N as *const _ as *const _,
                //                 g.st as *const _ as *const _,
                //             );
                //         }
                //     }
                //     RiOpacity(&WHITE);
                //     bind::obj_free(buzz);
                // };
                // RiTransformEnd();
            }
            RiWorldEnd();
        }
        RiFrameEnd();
    }
    RiEnd();
}

// const BLACK: RtColor = RtColor::new(0., 0., 0.);
const WHITE: RtColor = RtColor::new(1., 1., 1.);
