#!/bin/bash
#download youtube videos, crop & slices them.
rm -rf output slices sets
mkdir output slices sets

input="music-links.txt"
downloadn=0
while IFS= read -r line
do
	downloadn=$((downloadn+1))
	echo "download: ${downloadn}"
	youtube-dl "$line" -o "sets/download${downloadn}.wav" --audio-format "wav"
	duration=`ffprobe -i "sets/download${downloadn}.wav" -show_entries format=duration -v quiet -of csv="p=0"  | awk '{print int($1)}'`
	start=0
	end=30
	while [ $end -lt $duration ]
	do
		name=`echo "output/set${downloadn}_${start}_${end}.wav"`
		ffmpeg -ss $start -i "sets/download${downloadn}.wav" -to 30 $name -nostdin
		end=$((end+30))
		start=$((start+30))
	done
done < "$input"

python3 beat_slicer.py
rm -rf output sets