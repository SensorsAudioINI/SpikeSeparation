from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import wavfile
from scipy.signal import hilbert, butter, lfilter, freqz
from samplerate import resample
from librosa import stft
import sys

from SpikeSep import remove_trigger

pos = [10,6]
sample = 'A'
base_dir = '/Data/Dropbox/Shared ISCAS2017Submissions/RecordingsDungeon/'
prefix_filenames = '/Data/Dropbox/spike_separation_joy/recordings_dungeon/dungeon_concurrent_'

filename_id = '%d_%d_%s' %(pos[0], pos[1], sample)
filename_whisper1 = prefix_filenames + filename_id + '_1.wav'
filename_whisper3 = prefix_filenames + filename_id + '_3.wav'
filename_groundtruth1 = '/Data/Dropbox/spike_separation_joy/recordings_dungeon/edited%s1_T.wav' %sample
filename_groundtruth8 = '/Data/Dropbox/spike_separation_joy/recordings_dungeon/edited%s8_T.wav' %sample
fs, whisper1 = wavfile.read(filename_whisper1)
fs, whisper3 = wavfile.read(filename_whisper3)
# plt.plot(whisper1[:115000, 0])
# ipd.display(ipd.Audio(whisper1[:200000, 0], rate=fs))
TR = 10000 if sample in ['A', 'C'] and pos == [9, 7] else 18000
trigger_index = np.where(whisper1>TR)[0][0]
# print trigger_index
# trigger_index = 115000
whisper1 = whisper1[trigger_index + int(fs / 32):]
whisper3 = whisper3[trigger_index + int(fs / 32):]

# make the same length
shortest_size = np.amin((whisper1.shape[0], whisper3.shape[0]))
whisper1_shortened = whisper1[:shortest_size,:]
whisper3_shortened = whisper3[:shortest_size,:]
whisper = np.concatenate((whisper1_shortened,whisper3_shortened),1)

# gt
_, groundtruth1 = wavfile.read(filename_groundtruth1)
groundtruth1 = remove_trigger(groundtruth1)
_, groundtruth8 = wavfile.read(filename_groundtruth8)
groundtruth8 = remove_trigger(groundtruth8)

a = groundtruth1
b = np.array(whisper1[:, 1])

c = b[:]

plt.plot(a[:500])
plt.plot(b[:500])
plt.show()
plt.figure()
res_f = 500
r_a = resample(a, res_f / fs)
r_b = resample(c, res_f / fs)
print r_a.shape
print r_b.shape
plt.plot(r_a[:1000])
plt.plot(r_b[:1000])
plt.show()


