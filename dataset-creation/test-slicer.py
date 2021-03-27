import librosa
import soundfile as sf
import os
import re
import numpy as np 
import pyrubberband as rb

BEATS = 4
TARGET_TEMPO = 130
OFFSET = 800

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def open_file(filename):
	directory = "output/"+filename
	y, sr = librosa.load(directory)
	return y, sr

def get_beat(y,sr):
	onset_env = librosa.onset.onset_strength(y, sr=sr, aggregate=np.median)
	tempo, beat_frames = librosa.beat.beat_track(onset_envelope = onset_env, sr=sr, units="samples", start_bpm = TARGET_TEMPO)
	return tempo, beat_frames

def adjust_tempo(y, sr,tempo):
	time_strech_rate = TARGET_TEMPO / tempo 

	y = rb.pyrb.time_stretch(y, sr, time_strech_rate)
	return y

def make_slice(y, beat_frames, downbeats):
	start = int(round(len(downbeats)/2))
	slice_ = y[downbeats[start]-OFFSET: downbeats[start+1]-OFFSET]
	return slice_

def save_slice(slice_, filename,sr):
	new_name = "slices/slice"+filename
	sf.write(new_name, slice_, sr)

def get_downbeats(y,tempo, beat_frames,sr):
	measures = len(beat_frames) // BEATS	
	beat_frames = librosa.samples_to_frames(beat_frames)
	onset_env = librosa.onset.onset_strength(y, sr=sr, aggregate=np.median)
	beat_strengths = onset_env[beat_frames]
	measure_beat_strengths = beat_strengths[:measures * BEATS].reshape(-1, BEATS)
	beat_pos_strength = np.sum(measure_beat_strengths, axis=0)
	downbeat_pos = np.argmax(beat_pos_strength)
	full_measure_beats = beat_frames[:measures * BEATS].reshape(-1, BEATS)
	downbeat_frames = full_measure_beats[:, downbeat_pos]
	return librosa.frames_to_samples(downbeat_frames)

def main():
	folder = sorted_alphanumeric(os.listdir('output'))
	for filename in folder:
		#print("slicing: {}".format(i))
		y, sr = open_file(filename)
		tempo, beat_frames = get_beat(y, sr)

		if 120 <= tempo <= 145:

			y_new = adjust_tempo(y, sr, tempo)
			new_tempo, new_beat_frames = get_beat(y_new, sr)
			downbeats = get_downbeats(y_new, new_tempo, new_beat_frames, sr)
			slice_ = make_slice(y_new, beat_frames,downbeats)
			if 129 <= new_tempo <= 131:
				save_slice(slice_, filename, sr)
				print ("filename: ", filename, "- old-tempo: ", tempo, " - new-tempo: ",new_tempo,"samples: ", y_new.shape)


if __name__ == "__main__":
	main()
