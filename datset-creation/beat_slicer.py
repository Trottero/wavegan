import librosa
import soundfile as sf
import os

BEATS = 4

def crop_file(filename):
	directory = "output/"+filename
	y, sr = librosa.load(directory)
	tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="samples")
	start = int(round(len(beat_frames)/2))
	clip_audio = y[beat_frames[start]: beat_frames[start+BEATS-1]]
	new_name = "slices/slice"+filename
	sf.write(new_name, clip_audio, sr)

entries = os.listdir('output')

for i in entries:
	print("slicing: {}".format(i))
	crop_file(i)


