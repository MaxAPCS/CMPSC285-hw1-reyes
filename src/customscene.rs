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
                // Planet
                RiTransformBegin();
                {
                    RiColor(&RtColor::new(0.4, 0., 0.));
                    RiSurface(Some(STANDARD), null(), 0);
                    RiTranslate(2., 1., 16.);
                    RiSphere(4., -4., 4., 360.);
                }
                RiTransformEnd();

                RiTransformBegin();
                {
                    RiColor(&RtColor::from_element(1.));
                    RiOpacity(&RtColor::new(0.2, 0.2, 0.4));
                    RiSphere(8., -8., 8., 180.);
                }
                RiTransformEnd();
            }
            RiWorldEnd();
        }
        RiFrameEnd();
    }
    RiEnd();
}
