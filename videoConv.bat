cd "ref"

pause
ffmpeg -i 20230115090638_12_1_204.mp4 -codec copy -bsf: h264_mp4toannexb -f h264 20230115090638_12_1_204.h264
pause