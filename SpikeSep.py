from __future__ import division

import mir_eval
import numpy as np
from utils import es as es
from utils import itd as itd
from utils import prob as prob
from librosa import stft, istft
from scipy.io import wavfile
import pandas as pd

num_bins = 80
max_itd = 800e-6
num_channels = 64
relevant_channels = np.arange(0, 64)
basedir = '/Data/Dropbox/spike_separation_joy/recordings_dungeon/'
prefix_filenames = basedir + 'dungeon_concurrent_'


def get_priors():
    filename_priors = '/Data/Dropbox/spike_separation_joy/recordings_dungeon/priors_dungeon'
    priors = np.load(filename_priors + '.npy')
    itd_dict = itd.get_itd_dict(max_itd, num_bins) * 1e6
    return priors, itd_dict


def get_timestamps_from_filename(filename):
    # IN: a filename
    # OUT: decoded data from that location with timestamps in seconds

    # load aedat data
    timestamps, addresses = es.loadaerdat(filename + '.aedat')
    # print 'load data from ', filename

    timestamps = timestamps.astype('int64')
    disc = np.where(timestamps[:-1] - timestamps[1:] > 0)[0]
    # print "Detected trigggers [{}]".format(len(disc))
    if len(disc) > 1:
        if disc[1] - disc[0] > 10000:
            addresses = addresses[:disc[-1]]
            timestamps = timestamps[:disc[-1]]

    index_trigger = disc[0] + 1
    addresses = addresses[index_trigger:]
    timestamps = timestamps[index_trigger:]

    timestamps, ear_id, type_id = es.decode_ams1c(timestamps, addresses, return_type=True)
    channel_id = type_id % 64

    return timestamps, ear_id, type_id, channel_id


def calculate_itds(timestamps, ear_id, type_id, channel_id, return_itd_indices=True):
    # print 'calculating itds'
    # filter for certain channels specified in relevant_channels
    #     indices_channels = np.isin(channel_id, relevant_channels)
    #     timestamps = timestamps[indices_channels]
    #     ear_id = ear_id[indices_channels]
    #     type_id = type_id[indices_channels]

    # calculate itds for the filtered timestamps
    itds, itd_indices = itd.get_itds(timestamps, ear_id, type_id, return_itd_indices=return_itd_indices)
    timestamps = timestamps[itd_indices]
    channel_id = channel_id[itd_indices]
    ear_id = ear_id[itd_indices]
    type_id = type_id[itd_indices]

    return timestamps, ear_id, type_id, channel_id, itds


def assign_the_spikes(itds, sigma=2):
    # print 'applying hidden markov model'
    priors, _ = get_priors()
    # hidden markov model
    index_angles = np.array(
        [[0, -90], [1, -70], [2, -50], [3, -30], [4, -10], [5, 10], [6, 30], [7, 50], [8, 70], [9, 90]])
    num_angles = len(priors)
    initial_estimate = np.ones(num_angles) / num_angles  # all angles are a priori equally likely
    transition_probabilities = prob.get_transition_probabilities(index_angles,
                                                                 sigma=sigma)  # gaussian a priori probability of itds given a certain position
    itd_dict = itd.get_itd_dict(max_itd, num_bins)  # array holding the mean values of all itd bins
    estimates, argmax_estimates = prob.get_estimates(itds, initial_estimate, transition_probabilities, itd_dict, priors)

    return estimates, argmax_estimates


def remove_trigger(signal):
    trigger_index = np.where(signal > 20000)[0][-1]
    signal_trigger_removed = signal[trigger_index + 1:]
    return signal_trigger_removed


def multi_stft(x, frame_len, frame_step, window=np.hanning):
    s = np.array([stft(xx, frame_len, frame_step, window=window) for xx in x])
    return s

def estimate_fw_mapping(x1_spec):
    V, D, Rxx, LL = [], [], [], []

    for ii in range(x1_spec.shape[2]):  # Every frequency separately
        _R1 = x1_spec[:, :, ii].dot(np.conj(x1_spec[:, :, ii].T))
        Rxx.append(_R1)
        [_d, _v] = np.linalg.eig(_R1)
        idx = np.argsort(_d)[::-1]
        D.append(np.diag(_d[idx]))
        V.append(_v[:, idx])

    for v, d in zip(V, D):
        LL += [v[:, 0] * d[0, 0] / np.abs(d[0, 0])]

    return np.vstack(LL).T


def step_bf(x_spec, LL):

    S = np.einsum('...ca,...cb', x_spec, np.conj(x_spec)) / x_spec.shape[1]

    Rzz = np.array(S)

    _C = LL
    W = np.zeros((Rzz.shape[0], x_spec.shape[2])).astype('complex64')

    for i, r, ll0 in (zip(range(Rzz.shape[0]), Rzz, _C.T)):
        invR = np.linalg.pinv(r)
        invRL0 = invR.dot(ll0)
        J0 = 1.0 / ll0.T.dot(np.conj(invRL0))
        weight0 = invRL0 * J0
        W[i, :] = weight0
    return W


def break_small(x, l=5):
    r = np.zeros_like(x)
    for i in range(len(x) - l):
        if np.sum(x[i:i+l]) == l:
            r[i:i+l] = 1
    return r


def calculate_best(sample, pos, fs=24000, use_cached=False):
    SIR = []
    SDR = []
    SAR = []
    if not use_cached:
        for CH in [0, 1, 2, 3]:
            for W in [1, 3]:
                init1 = 0
                init0 = int(30 * fs) if sample == 'A' else int(13 * fs)
                init2 = int(40 * fs) if sample == 'A' else int(26 * fs)

                s2 = 'B' if sample == 'A' else 'A'
                TR = 10000 if s2 in ['A', 'C'] and pos == [9, 7] else 18000
                filename_id = '%d_%d_%s' % (pos[0], pos[1], s2)
                filename_whisper1 = prefix_filenames + filename_id + '_{}.wav'.format(W)
                fs, whisper1 = wavfile.read(filename_whisper1)
                trigger_index = np.where(whisper1 > TR)[0][0]
                whisperB = whisper1[trigger_index + int(fs / 32):, CH].astype('float32')

                s3 = 'D' if sample == 'A' else 'B'
                TR = 10000 if s3 in ['A', 'C'] and pos == [9, 7] else 18000
                filename_id = '%d_%d_%s' % (pos[0], pos[1], s3)
                filename_whisper1 = prefix_filenames + filename_id + '_{}.wav'.format(W)
                fs, whisper1 = wavfile.read(filename_whisper1)
                trigger_index = np.where(whisper1 > TR)[0][0]
                whisperC = whisper1[trigger_index + int(fs / 32):, CH].astype('float32')

                whisperB = whisperB[init0:init2]
                whisperC = whisperC[init0:init2]

                filename_groundtruth1 = '/Data/Dropbox/spike_separation_joy/recordings_dungeon/edited%s1_T.wav' % sample
                filename_groundtruth8 = '/Data/Dropbox/spike_separation_joy/recordings_dungeon/edited%s8_T.wav' % sample
                # ground truth
                fs, groundtruth1 = wavfile.read(filename_groundtruth1)
                groundtruth1 = remove_trigger(groundtruth1)
                fs, groundtruth8 = wavfile.read(filename_groundtruth8)
                groundtruth8 = remove_trigger(groundtruth8)
                groundtruth11 = groundtruth1[init0:init2]
                groundtruth88 = groundtruth8[init0:init2]

                end = 100000

                pp = np.correlate(whisperB[init1:init1 + end], groundtruth88[init1:init1 + end], 'full')
                j1 = np.argmax(pp) - end

                pp = np.correlate(whisperC[init1:init1 + end], groundtruth11[init1:init1 + end], 'full')
                j0 = np.argmax(pp) - end

                # print "{} || {}".format(j0, j1)

                if j0 > 0:
                    whisperC = whisperC[j0:]
                else:
                    groundtruth11 = groundtruth11[abs(j0):]

                if j1 > 0:
                    whisperB = whisperB[j1:]
                else:
                    groundtruth88 = groundtruth88[abs(j1):]

                min_len = np.min([len(whisperB), len(whisperC), len(groundtruth11), len(groundtruth88)])
                whisperB = whisperB[:min_len]
                whisperC = whisperC[:min_len]
                groundtruth11 = groundtruth11[:min_len]
                groundtruth88 = groundtruth88[:min_len]

                # plt.plot(groundtruth88 * 2)
                # plt.plot(whisperB)
                # plt.figure()
                # plt.plot(groundtruth11 * 2)
                # plt.plot(whisperC)

                sdr_b, sir_b, sar_b, _ = mir_eval.separation.bss_eval_sources(np.array([groundtruth11, groundtruth88]),
                                                                              np.array([whisperC, whisperB]))

                SIR.append(sir_b)
                SAR.append(sar_b)
                SDR.append(sdr_b)

        sir_b = np.max(np.array(SIR), axis=0)
        sdr_b = np.max(np.array(SDR), axis=0)
        sar_b = np.max(np.array(SAR), axis=0)

    else:
        df = pd.read_csv('baseline_best.csv')
        idx = np.where(np.logical_and(np.logical_and(df['Conf1'] == pos[1], df['Conf0'] == pos[0]), df['Type'] == sample))[0][0]
        sir_b = df['SIR1'][idx], df['SIR2'][idx]
        sdr_b = df['SDR1'][idx], df['SDR2'][idx]
        sar_b = df['SAR1'][idx], df['SAR2'][idx]

    return sir_b, sdr_b, sar_b


def triggers(sample, fs, frame_step, jit=0):
    r = fs / frame_step
    t0 = 0 + jit
    t1 = int(12.78125 * r) + jit
    t2 = int(26.56250 * r) + jit
    t3 = int(40.34375 * r) + jit
    t4 = int(50.12500 * r) + jit
    
    if sample == 'A':
        return slice(t0, t1), slice(t1, t2), [slice(t3, t4)]

    elif sample == 'B':
        return slice(t1, t2), slice(t2, t3), [slice(t3, t4)]

    elif sample == 'C':
        return slice(t0, t1), slice(t2, t3), [slice(t3, t4)]

    elif sample == 'D':
        return slice(t2, t3), slice(t1, t2), [slice(t3, t4)]

    # if sample == 'A':
    #     return slice(t0, t1), slice(t1, t2), [slice(t2, t3), slice(t3, t4)]
    #
    # elif sample == 'B':
    #     return slice(t1, t2), slice(t2, t3), [slice(t0, t1), slice(t3, t4)]
    #
    # elif sample == 'C':
    #     return slice(t0, t1), slice(t2, t3), [slice(t1, t2), slice(t3, t4)]
    #
    # elif sample == 'D':
    #     return slice(t2, t3), slice(t1, t2), [slice(t0, t1), slice(t3, t4)]

