scenes_dir="media/videos/process_anim/1080p60"

for FILE in $scenes_dir/*.mp4; 
do echo -e "file $FILE" >> files.txt;
done

ffmpeg -f concat -safe 0 -i files.txt -c copy final.mp4

rm files.txt
