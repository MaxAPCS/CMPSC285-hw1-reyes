# CMPSC285 Homework 1 - REYES Renderer
### Build
```
cargo build --release
```
Artifact can be found at [target/release/reyes](target/release/reyes).

### Execution
Must be run with assets/ in working directory.
```
reyes                list scenes
reyes <scene>        one scene, by name
reyes CustomScene    the custom scene
reyes test           every test scene
reyes intermediate   every intermediate scene
reyes cinematic      every cinematic scene
reyes stills         everything but the two movies
reyes all            everything
```
The two scenes that render a movie (`SimpleAnimationMovie` and
`WallEMovie`) write numbered frames. Turn them into an MP4 with ffmpeg:
```
ffmpeg -framerate 24 -i renders/1_TestScenes/SimpleAnimation/SimpleAnimation_%02d.png \
       -c:v libx264 -pix_fmt yuv420p -crf 18 renders/1_TestScenes/SimpleAnimation.mp4
ffmpeg -framerate 24 -i renders/3_CinematicScenes/WallEMovie/WallEMovie_%02d.jpg \
       -c:v libx264 -pix_fmt yuv420p -crf 18 renders/3_CinematicScenes/WallEMovie.mp4
```
NB: Image output is performed with image crate's built in conversions, so color space linearity may be different.

### Results
Benchmarked on an Intel i9-13900KF with 5600MHz DDR5 RAM

| Time (s)   | Scene                   |
|------------|-------------------------|
|    0.079664 | FirstPrimitive_sphere   |
|    0.023983 | FirstPrimitive_cylinder |
|    0.023875 | FirstPrimitive_cone     |
|    0.050186 | FirstPrimitive_torus    |
|    0.401337 | OrthoVsPersp_ortho      |
|    0.112923 | OrthoVsPersp_persp      |
|    0.269525 | SimpleAnimation         |
|   26.583604 | SimpleAnimationMovie    |
|    0.255202 | AliasingTest_1x1        |
|    0.502062 | AliasingTest_4x4        |
|    0.840669 | DiceRateTest            |
|    0.569658 | QuadricZoo              |
|    0.296246 | Tunnel                  |
|    0.278345 | ShaderTest              |
|    0.805270 | UVTest                  |
|    0.165760 | Earth                   |
|    0.765320 | DisplacementTest        |
|    7.453686 | BumpVsDisplacement      |
|    0.266320 | Tornado                 |
|    0.159887 | TransparencyTest        |
|    0.970510 | TransparencyOrder       |
|    0.639364 | PolygonTest             |
|    4.528120 | ShadowTest              |
|    1.618921 | NormalTest              |
|    0.727940 | Rocket                  |
|    5.487877 | Explorer                |
|   84.174365 | McQueen                 |
|   43.721284 | Buzz                    |
|   80.902341 | WallE                   |
|   25.816528 | Gumbo                   |
|  406.242271 | RadiatorSprings         |
|   39.180757 | AndysRoom               |
| 1732.911775 | WallEMovie              |

### Footnotes
**Extra Credit:**
- Per-Primitive Parallelization: Shaders, Projection, Normal Recomputation, Buffer Rasterization

**Model Sources:**
- Provided Models.

AI was used for debugging, all code written by hand.