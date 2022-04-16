#### video to gif ####
# accelerate the video:
ffmpeg -i input.mp4 -filter:v "setpts=0.3*PTS" output.mp4
# compress to gif with optimization
ffmpeg -y -i output.mp4 -filter_complex "fps=5,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=32[p];[s1][p]paletteuse=dither=bayer" output.gif