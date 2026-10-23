use std::ptr::null;

use crate::{Scene, ri_api::*, shaders::*};

pub static SCENE: Scene = Scene {
    name: "CustomScene",
    tier: "cinematic",
    function: Function,
};

#[allow(non_snake_case)]
unsafe extern "C" fn Function() {
    function();
}

#[inline]
fn function() {
    RiBegin(null());
    {
        ri_display("CustomScene.png".to_string(), false);
        RiFormat(1920, 1080);
        RiPixelSamples(4, 4);
        ri_projection("perspective", Some(35.));
        RiFrameBegin(0);
        {
            RiWorldBegin();
            {
                RiTransformBegin();
                {
                    RiSurface(Some(STANDARD), null(), 0);
                    RiTranslate(2., -0.5, 4.);
                    RiSphere(4., -4., 4., 360.);
                }
                RiTransformEnd();
            }
            RiWorldEnd();
        }
        RiFrameEnd();
    }
    RiEnd();
}
