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
reyes custom         the custom scene
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

### Footnotes
**Extra Credit:**
- Parallelization: All shaders run in parallel. Every scene executes faster.

**Model Sources:**
- Provided Models.

AI was used for debugging, all code written by hand.