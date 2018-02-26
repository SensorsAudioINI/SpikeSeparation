from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import wavfile
from scipy.signal import hilbert, butter, lfilter, freqz
from samplerate import resample
from scipy.io import loadmat

# low pass
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


_, source1 = wavfile.read('/Data/Dropbox/Workshops/Telluride 2016/ori_f_rm_special_tsp.wav')
fs, source2 = wavfile.read('/Data/Dropbox/Workshops/Telluride 2016/ori_m_rm_special_tsp.wav')

min_len = min([len(source1), len(source2)])
source1 = source1[:min_len]
source2 = source2[:min_len]

res_f = 200
res_source1 = resample(source1, res_f / fs)
res_source2 = resample(source2, res_f / fs)

print len(source1)
print len(res_source1)

Wn = 20
env1 = butter_lowpass_filter(np.abs(hilbert(res_source1)), Wn, res_f)
env2 = butter_lowpass_filter(np.abs(hilbert(res_source2)), Wn, res_f)

# envs = loadmat('/Data/Dropbox/Workshops/Telluride 2016/envs.mat')
# env1 = np.squeeze(envs['env1'])
# env2 = np.squeeze(envs['env2'])

N = 2  # number of sources
M = len(env1)  # timesteps
J = 4  # number of mics
# Create random data
original_S = np.zeros((N, M))
original_S[0, :] = env1 / np.max(env1)
original_S[1, :] = env2 / np.max(env2)
# original_S = envs['original_S']
A = 5 * np.random.rand(J, N)  # random mixing matrix
Y = A.dot(original_S)
zm_original_S = original_S - np.mean(original_S, keepdims=True, axis=1)

S_init = np.abs(Y[:N, :M])

U, SIGMA, V = np.linalg.svd(Y, full_matrices=True)
V = V.T

V_not = V[:, :N]

oneM = np.ones((M, 1))

S_prev = S_init

iteration = 1

ex = False
while not ex:

    S_not = 1.0 / M * S_prev.dot(oneM.dot(oneM.T))

    C_s = (S_prev - S_not).dot((S_prev - S_not).T)

    D1_i = np.linalg.inv(np.diag(np.diag(C_s)))

    D2 = np.diag(np.diag((D1_i.dot(C_s)) ** 2))

    a = S_prev
    b = S_not.dot(S_prev.T).dot(D1_i).dot(S_prev) + S_prev.dot(S_prev.T).dot(D1_i).dot(S_not) + D2.dot(S_prev)
    c = S_not.dot(S_prev.T).dot(D1_i).dot(S_not) + S_prev.dot(S_prev.T).dot(D1_i).dot(S_prev) + D2.dot(S_not)

    S_temp = a * b / c

    S_prev = np.clip(S_temp.dot(V_not.dot(V_not.T)), 0, np.Inf)

    norm_S_prev = S_prev / np.max(S_prev, axis=1, keepdims=True)
    zm_S_prev = norm_S_prev - np.mean(norm_S_prev, axis=1, keepdims=True)

    corr11 = np.corrcoef(zm_original_S[0, :], zm_S_prev[0, :])[0, 1]
    corr12 = np.corrcoef(zm_original_S[0, :], zm_S_prev[1, :])[0, 1]
    corr21 = np.corrcoef(zm_original_S[1, :], zm_S_prev[0, :])[0, 1]
    corr22 = np.corrcoef(zm_original_S[1, :], zm_S_prev[1, :])[0, 1]
    # print "{} {} {} {}".format(corr11, corr12, corr21, corr22)
    iteration += 1

    if (corr11 > 0.95 and corr22 > 0.95) or (corr12 > 0.95 and corr21 > 0.95):
        ex = True
        print 'Took {} iterations'.format(iteration)

