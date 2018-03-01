from __future__ import division

import mir_eval
import numpy as np
from utils import es as es
from utils import itd as itd
from utils import prob as prob
from librosa import stft, istft
from scipy.io import wavfile
import pandas as pd
from gccNMFFunctions import *
import os.path
import pickle
import sys
from scipy.signal import hilbert, butter, lfilter, freqz
from samplerate import resample

num_bins = 80
max_itd = 800e-6
num_channels = 64
relevant_channels = np.arange(0, 64)
basedir = '/Data/Dropbox/spike_separation_joy/recordings_dungeon/'
prefix_filenames = basedir + 'dungeon_concurrent_'


def low_pass(x, ll=50):
    r = np.zeros_like(x)
    for i in range(len(x) - ll):
        r[i] = np.mean(x[i:i + ll])
    return r


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


class SAD(object):
    def __init__(self, pos, sample, fs=24000, frame_len=2048, frame_step=512, break_w_ms=200, jit=None):

        self.sample = sample
        self.pos = pos
        self.frame_len = frame_len
        self.frame_step = frame_step
        self.jit = int(frame_len / frame_step / 2) if jit is None else jit
        self.fs = fs
        self.break_w = int(np.ceil(break_w_ms / 1000.0 * self.fs / frame_step))

        filename_id = '{}_{}_{}'.format(pos[0], pos[1], sample)
        filename_cochlea = prefix_filenames + filename_id
        filename_whisper1 = prefix_filenames + filename_id + '_1.wav'
        filename_whisper3 = prefix_filenames + filename_id + '_3.wav'
        filename_groundtruth1 = basedir + '/edited%s1_T.wav' % sample
        filename_groundtruth8 = basedir + '/edited%s8_T.wav' % sample

        # gt
        _, self.gt1 = wavfile.read(filename_groundtruth1)
        self.gt1 = remove_trigger(self.gt1)
        _, self.gt8 = wavfile.read(filename_groundtruth8)
        self.gt8 = remove_trigger(self.gt8)

        # spikes
        self.timestamps, self.ear_id, self.type_id, self.channel_id = get_timestamps_from_filename(filename_cochlea)

        # whisper
        fs, whisper1 = wavfile.read(filename_whisper1)
        fs, whisper3 = wavfile.read(filename_whisper3)
        TR = 10000 if sample in ['A', 'C'] and pos == [9, 7] else 18000

        trigger_index = np.where(whisper1 > TR)[0][0]
        whisper1 = whisper1[trigger_index + int(fs / 32):].astype('float32')
        whisper3 = whisper3[trigger_index + int(fs / 32):].astype('float32')

        # make the same length
        shortest_size = np.amin((whisper1.shape[0], whisper3.shape[0]))
        whisper1_shortened = whisper1[:shortest_size, :]
        whisper3_shortened = whisper3[:shortest_size, :]
        self.whisper = np.concatenate((whisper1_shortened, whisper3_shortened), 1)
        self.x_spec = np.transpose(multi_stft(self.whisper.T, self.frame_len, self.frame_step, window=np.hanning),
                                   (0, 2, 1))

        # SNR
        slice_spk1, slice_spk2, _ = triggers(sample, self.fs, 1)

        spk1 = []
        spk2 = []
        for i in range(8):
            spk1 += [10 * np.log10(np.sum(self.whisper[slice_spk1, i] ** 2))]
            spk2 += [10 * np.log10(np.sum(self.whisper[slice_spk2, i] ** 2))]

        self.SNR = np.mean(spk1) - np.mean(spk2)

    def apply_sad(self, sad1, sad2, val_ch):
        m2 = stft(self.gt8, self.frame_len, self.frame_step, window=np.hanning).T
        m1 = stft(self.gt1, self.frame_len, self.frame_step, window=np.hanning).T
        m1_sum = np.sum(np.log(m1 + 1), axis=-1)
        m2_sum = np.sum(np.log(m2 + 1), axis=-1)
        SAD1 = np.array(0.8 * m1_sum > m2_sum)
        if len(SAD1) > len(sad1):
            R = int(np.floor(self.x_spec.shape[1] / len(sad1)))
            sad1 = np.repeat(sad1, R)
            sad2 = np.repeat(sad2, R)
        else:
            R = int(np.ceil(len(sad1) / len(SAD1)))
            sad1 = sad1[::R]
            sad2 = sad2[::R]
        min_len = min([len(sad1), len(SAD1)])
        sad1 = sad1[:min_len]
        sad2 = sad2[:min_len]

        gt_vad1, gt_vad2, vad_slice = triggers(self.sample, self.fs, self.frame_step, self.jit)
        gt_s1, gt_s2, spec_slice = triggers(self.sample, self.fs, self.frame_step)
        x_spec = self.x_spec[:, :len(sad1), :]

        recons = []
        for vad in [sad1, sad2]:
            vad = np.expand_dims(np.expand_dims(break_small(vad, self.break_w), 0), -1)
            xx = []
            for v_sl, x_sl in zip(vad_slice, spec_slice):
                xx += [x_spec[val_ch, x_sl] * vad[:, v_sl]]
            ll1 = estimate_fw_mapping(np.concatenate(xx, axis=1))
            x_spec_b = x_spec[val_ch].T
            w = step_bf(x_spec_b, ll1)
            recons.append(np.einsum('ab,acb->ca', np.conj(w), x_spec_b))
        recons = [istft(x.T, self.frame_step, window=np.hanning) for x in recons]

        return recons

    def emp_limit_LCMV(self, val_ch):
        _filename = 'cache/emp_lim_{}_{}_{}_{}_{}.pkl'.format(self.pos[0], self.pos[1], self.sample, self.frame_len,
                                                              self.frame_step)
        if not os.path.isfile(_filename):
            recons = []
            gt_s1, gt_s2, spec_slice = triggers(self.sample, self.fs, self.frame_step, 4)

            # EMP LIMIT
            for x_sl in [gt_s1, gt_s2]:
                xx = self.x_spec[val_ch, x_sl]
                ll1 = estimate_fw_mapping(xx)
                x_spec_b = self.x_spec[val_ch].T
                w = step_bf(x_spec_b, ll1)
                recons.append(np.einsum('ab,acb->ca', np.conj(w), x_spec_b))

            recons = [istft(x.T, self.frame_step, window=np.hanning) for x in recons]
            pickle.dump(recons, open(_filename, 'w'))
            return recons
        else:
            print "Using cached EMP..."
            return pickle.load(open(_filename, 'r'))

    def evaluate_bss(self, recons, save_name=None):
        init1 = 0
        init0 = int(30 * self.fs) if self.sample == 'A' else int(16 * self.fs)
        init2 = int(40 * self.fs) if self.sample == 'A' else int(26 * self.fs)
        # init_1 = int(25 * fs)
        recons0 = recons[0][init0:init2]
        recons1 = recons[1][init0:init2]
        groundtruth11 = self.gt1[init0:]
        groundtruth88 = self.gt8[init0:]

        end = 100000

        pp = np.correlate(recons1[init1:init1 + end], groundtruth88[init1:init1 + end], 'full')
        j1 = np.argmax(pp) - end

        pp = np.correlate(recons0[init1:init1 + end], groundtruth11[init1:init1 + end], 'full')
        j0 = np.argmax(pp) - end

        if j0 > 0:
            recons0 = recons0[j0:]
        else:
            groundtruth11 = groundtruth11[abs(j0):]

        if j1 > 0:
            recons1 = recons1[j1:]
        else:
            groundtruth88 = groundtruth88[abs(j1):]

        min_len = np.min([len(recons0), len(recons1), len(groundtruth11), len(groundtruth88)])
        recons0 = recons0[:min_len]
        recons1 = recons1[:min_len]
        groundtruth11 = groundtruth11[:min_len]
        groundtruth88 = groundtruth88[:min_len]

        if save_name is not None:
            wavfile.write('output_wavs/{}_{}_{}_1.wav'.format(save_name, self.pos, self.sample), self.fs,
                          recons0 / np.max(np.abs(recons0)))
            wavfile.write('output_wavs/{}_{}_{}_2.wav'.format(save_name, self.pos, self.sample), self.fs,
                          recons1 / np.max(np.abs(recons1)))
            wavfile.write('output_wavs/{}_{}_{}_1_gt.wav'.format(save_name, self.pos, self.sample), self.fs,
                          groundtruth11 / np.max(np.abs(groundtruth11)))
            wavfile.write('output_wavs/{}_{}_{}_2_gt.wav'.format(save_name, self.pos, self.sample), self.fs,
                          groundtruth88 / np.max(np.abs(groundtruth88)))

        return mir_eval.separation.bss_eval_sources(np.array([groundtruth11, groundtruth88]),
                                                    np.array([recons0, recons1]))

    def get_MNICASAD(self, cut_f=20, res_f=188, th=0.9):

        env1 = butter_lowpass_filter(
            np.abs(hilbert(self.gt1, N=int(2 ** np.ceil(np.log2(len(self.gt1)))))), cut_f, self.fs)
        env2 = butter_lowpass_filter(
            np.abs(hilbert(self.gt8, N=int(2 ** np.ceil(np.log2(len(self.gt8)))))), cut_f, self.fs)

        env1 = env1[:len(self.gt1)]
        env2 = env2[:len(self.gt1)]

        env1 = resample(env1, res_f / self.fs)
        env2 = resample(env2, res_f / self.fs)

        # stfts & VADs
        VAD1 = np.array(0.8 * env1 > env2)
        VAD2 = np.array(0.8 * env2 > env1)

        Y = np.array([resample(
            butter_lowpass_filter(np.abs(hilbert(x, N=int(2 ** np.ceil(np.log2(len(x)))))), cut_f, self.fs), res_f / self.fs)
            for x in self.whisper.T])

        ori = np.array([env1, env2])
        Y = Y[:, :ori.shape[1]]

        pp = np.correlate(Y[0][:1000], ori[0][:1000], 'full')
        j0 = np.argmax(pp) - 1000

        if j0 > 0:
            Y = Y[:, j0:]
            ori = ori[:, :-j0]
        else:
            ori = ori[:, abs(j0):]
            Y = Y[:, :-abs(j0)]

        # m-nica and vad
        est, o = m_nica(Y, ori, verbose=True, max_iter=500, th=0.9, patience=10)

        sad1 = np.float32(est[0] * th > est[1])
        sad2 = np.float32(est[1] * th > est[0])

        return {'sad1': sad1, 'sad2': sad2, 'first': 0, 'second': 0, 'corr1': 0, 'corr2': 0}

    def get_GCCSAD(self, frame_len=8192, frame_step=2048, th=0.95, lp=5):

        stereo_signal = np.array([butter_lowpass_filter(self.whisper[:, 0], 11000, self.fs),
                                  butter_lowpass_filter(self.whisper[:, 1], 11000, self.fs)]).T
        complexMixtureSpectrogram = computeComplexMixtureSpectrogram(stereo_signal.T, frame_len, frame_step,
                                                                     np.hanning)
        numChannels, numFrequencies, numTime = complexMixtureSpectrogram.shape
        frequenciesInHz = getFrequenciesInHz(self.fs, numFrequencies)

        spectralCoherenceV = complexMixtureSpectrogram[0] * complexMixtureSpectrogram[1].conj() \
                             / abs(complexMixtureSpectrogram[0]) / abs(complexMixtureSpectrogram[1])
        angularSpectrogram = getAngularSpectrogram(spectralCoherenceV, frequenciesInHz,
                                                   0.5, 128)
        angularSpectrogram = np.nan_to_num(angularSpectrogram)
        meanAngularSpectrum = mean(angularSpectrogram, axis=-1)
        targetTDOAIndexes = estimateTargetTDOAIndexesFromAngularSpectrum(meanAngularSpectrum, 0.5, 128, 3)

        B = low_pass(angularSpectrogram[targetTDOAIndexes[0], :], ll=lp)
        A = low_pass(angularSpectrogram[targetTDOAIndexes[-1], :], ll=lp)
        A -= np.min(A)
        B -= np.min(B)
        A /= np.max(A)
        B /= np.max(B)

        sad1 = np.float32(A * th > B)
        sad2 = np.float32(B * th > A)

        return {'sad1': sad1, 'sad2': sad2, 'first': 0, 'second': 0, 'corr1': 0, 'corr2': 0}

    def get_SOSAD(self, frame_len=8192, frame_step=2048, th=0.9):

        _filename = 'cache/cached_{}_{}_{}_{}_{}_{}.pkl'.format(self.pos[0], self.pos[1], self.sample, frame_len,
                                                                frame_step, th)

        if not os.path.isfile(_filename):

            timestamps, ear_id, type_id, channel_id, itds = calculate_itds(self.timestamps, self.ear_id, self.type_id,
                                                                           self.channel_id,
                                                                           return_itd_indices=True)

            estimates, argmax_estimates = assign_the_spikes(itds, sigma=45)

            frame_len_s = frame_len / self.fs
            frame_step_s = frame_step / self.fs

            mag1 = stft(self.gt1, frame_len, frame_step, window=np.hanning)
            e1 = np.sum(np.log(np.abs(mag1) + 1), axis=0)
            mag8 = stft(self.gt8, frame_len, frame_step, window=np.hanning)
            e8 = np.sum(np.log(np.abs(mag8) + 1), axis=0)
            r_ch = range(10, 20)

            scores = []
            envs = []
            for idx in range(estimates.shape[1]):
                a = timestamps[argmax_estimates == idx]
                f_ch = channel_id[argmax_estimates == idx]
                summed_prob = np.sum(estimates[:, idx]) / len(timestamps)
                if len(a) == 0:
                    envs.append(np.zeros((len(e1),)))
                    continue
                a = a[np.array([k in r_ch for k in f_ch])]
                ts = np.arange(0, frame_step_s * (len(e1) + 1), frame_step_s)
                c = [len(a[(a > t) & (a < (t + frame_len_s))]) for i, t in enumerate(ts[:-1])]

                _score = summed_prob * len(a)
                scores.append(_score)
                envs.append(c)

            corr_envs = np.corrcoef(envs)
            weights = 0.005 ** (corr_envs[np.argmax(scores)] + 1)

            first = np.argmax(scores) + 1
            second = np.argmax(np.log(scores * weights)) + 1

            if first != second:
                pass
            else:
                third = np.argsort(np.log(scores * weights))[-2] + 1
                second = third

            estimated_env_1 = np.log(np.array(envs[int(first) - 1]) + 1)
            estimated_env_2 = np.log(np.array(envs[second - 1]) + 1)

            corr_env_1 = np.corrcoef(estimated_env_1, e1)[0, 1]
            corr_env_2 = np.corrcoef(estimated_env_2, e8)[0, 1]

            sad1 = np.float32(estimated_env_1 - th * estimated_env_1 > estimated_env_2)
            sad2 = np.float32(estimated_env_2 - th * estimated_env_2 > estimated_env_1)

            res = {'sad1': sad1, 'sad2': sad2,
                   'corr1': corr_env_1, 'corr2': corr_env_2,
                   'env1': estimated_env_1, 'env2': estimated_env_2,
                   'first': first, 'second': second}

            pickle.dump(res, open(_filename, 'w'))

            return res
        else:
            print "Using cached...SOSAD"
            return pickle.load(open(_filename, 'r'))


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
        if np.sum(x[i:i + l]) == l:
            r[i:i + l] = 1
    return r


def calculate_best(sample, pos, fs=24000, use_cached=False):
    SIR = []
    SDR = []
    SAR = []
    if not use_cached:
        for CH in [0, 1, 2, 3]:
            for W in [1, 3]:
                init1 = 0
                init0 = int(30 * fs) if sample == 'A' else int(16 * fs)
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
        idx = \
            np.where(
                np.logical_and(np.logical_and(df['Conf1'] == pos[1], df['Conf0'] == pos[0]), df['Type'] == sample))[0][
                0]
        sir_b = df['SIR1'][idx], df['SIR2'][idx]
        sdr_b = df['SDR1'][idx], df['SDR2'][idx]
        sar_b = df['SAR1'][idx], df['SAR2'][idx]

    return sir_b, sdr_b, sar_b


def triggers(sample, fs, frame_step, jit=0, mode='last'):
    r = fs / frame_step
    t0 = 0 + jit
    t1 = int(12.78125 * r) + jit
    t2 = int(26.56250 * r) + jit
    t3 = int(40.34375 * r) + jit
    t4 = int(50.12500 * r) + jit
    if mode == 'last':
        if sample == 'A':
            return slice(t0, t1), slice(t1, t2), [slice(t3, t4)]

        elif sample == 'B':
            return slice(t1, t2), slice(t2, t3), [slice(t3, t4)]

        elif sample == 'C':
            return slice(t0, t1), slice(t2, t3), [slice(t3, t4)]

        elif sample == 'D':
            return slice(t2, t3), slice(t1, t2), [slice(t3, t4)]

    if mode == 'first':
        if sample == 'A':
            return slice(t0, t1), slice(t1, t2), [slice(t2, t3)]

        elif sample == 'B':
            return slice(t1, t2), slice(t2, t3), [slice(t0, t1)]

        elif sample == 'C':
            return slice(t0, t1), slice(t2, t3), [slice(t1, t2)]

        elif sample == 'D':
            return slice(t2, t3), slice(t1, t2), [slice(t0, t1)]

    if mode == 'both':
        if sample == 'A':
            return slice(t0, t1), slice(t1, t2), [slice(t2, t3), slice(t3, t4)]

        elif sample == 'B':
            return slice(t1, t2), slice(t2, t3), [slice(t0, t1), slice(t3, t4)]

        elif sample == 'C':
            return slice(t0, t1), slice(t2, t3), [slice(t1, t2), slice(t3, t4)]

        elif sample == 'D':
            return slice(t2, t3), slice(t1, t2), [slice(t0, t1), slice(t3, t4)]


def m_nica(y, x, max_iter=100, n=2, th=0.95, verbose=False, patience=10):
    zm_x = x - np.mean(x, axis=1, keepdims=True)
    t = y.shape[-1]
    S_init = np.abs(y[:n, :])
    _, _, V = np.linalg.svd(y, full_matrices=True)
    V = V.T
    V_not = V[:, :n]
    oneM = np.ones((t, 1))
    S_prev = S_init
    iteration = 1
    ex = 0
    pat_count = 0
    last_best = 0
    for i in range(max_iter):

        S_not = 1.0 / t * (S_prev).dot(oneM.dot(oneM.T))

        C_s = (S_prev - S_not).dot((S_prev - S_not).T)

        D1_i = np.linalg.inv(np.diag(np.diag(C_s)))

        D2 = np.diag(np.diag((D1_i.dot(C_s)) ** 2))

        S_temp = S_prev * \
                 (S_not.dot(S_prev.T).dot(D1_i).dot(S_prev) + S_prev.dot(S_prev.T).dot(D1_i).dot(S_not) + D2.dot(
                     S_prev)) / \
                 (S_not.dot(S_prev.T).dot(D1_i).dot(S_not) + S_prev.dot(S_prev.T).dot(D1_i).dot(S_prev) + D2.dot(S_not))

        S_prev = np.clip(S_temp.dot(V_not).dot(V_not.T), 0, np.Inf)
        norm_S_prev = S_prev / np.max(S_prev, axis=1, keepdims=True)
        zm_S_prev = norm_S_prev - np.mean(norm_S_prev, axis=1, keepdims=True)

        corrs = []
        corrs.append([np.corrcoef(zm_S_prev[0, :], zm_x[0, :])[0, 1], np.corrcoef(zm_S_prev[1, :], zm_x[1, :])[0, 1]])
        corrs.append([np.corrcoef(zm_S_prev[1, :], zm_x[0, :])[0, 1], np.corrcoef(zm_S_prev[0, :], zm_x[1, :])[0, 1]])

        arg = np.argmax([np.sum(corrs[0]), np.sum(corrs[1])])
        if np.sum(corrs[arg]) - 0.01 < last_best:
            pat_count += 1
            if pat_count == patience:
                break
        else:
            last_best = np.sum(corrs[arg])

        iteration += 1
        if verbose:
            sys.stdout.write("\rIteration %d: %.4f || %.4f " % (iteration, corrs[arg][0], corrs[arg][1]))
        if corrs[0][0] > th and corrs[0][1] > th:
            ex = 2
            break
        if corrs[1][0] > th and corrs[1][1] > th:
            ex = 1
            break

    est1 = norm_S_prev[0, :]
    est2 = norm_S_prev[1, :]

    if ex == 1:
        ori1 = x[0, :]
        ori2 = x[1, :]
    else:
        ori1 = x[1, :]
        ori2 = x[0, :]

    return np.array([est1, est2]), np.array([ori1, ori2])