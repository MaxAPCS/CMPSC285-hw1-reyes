```
reyes                list scenes
reyes <scene>        one scene, by name
reyes test           every test scene
reyes intermediate   every intermediate scene
reyes cinematic      every cinematic scene
reyes stills         everything but the two movies
reyes all            everything
```
The two scenes that render a movie (`SimpleAnimationMovie` and
`WallEMovie`) write numbered frames. Turn them into an MP4 with ffmpeg (the
same command is in the comment at the top of each scene):

    ffmpeg -framerate 24 -i renders/1_TestScenes/SimpleAnimation/SimpleAnimation_%02d.png \
           -c:v libx264 -pix_fmt yuv420p -crf 18 renders/1_TestScenes/SimpleAnimation.mp4
    ffmpeg -framerate 24 -i renders/3_CinematicScenes/WallEMovie/WallEMovie_%02d.jpg \
           -c:v libx264 -pix_fmt yuv420p -crf 18 renders/3_CinematicScenes/WallEMovie.mp4

Both scenes have a background behind everything, so every frame is opaque
and the movie needs nothing more. (An MP4 has no alpha: frames with
transparent parts would need `-vf premultiply=inplace=1` first, or ffmpeg
drops the alpha without applying it and the edges turn jagged.)
