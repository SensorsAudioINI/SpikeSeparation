import mir_eval
import numpy as np
from librosa import stft, istft
from scipy.io import wavfile
from scipy.signal import butter, lfilter

from SpikeSep import triggers, multi_stft, remove_trigger, estimate_fw_mapping, step_bf, break_small
from gccNMFFunctions import *


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


CHS = [
    np.array([0, 1, 2, 3, 4, 5, 6]),  # [10,1] # 0
    np.array([0, 1, 2, 3, 4, 5, 6]),  # [10,6] # 1
    np.array([0, 1, 2, 3, 5, 6]),  # [1,8] # 1
    np.array([0, 1, 2, 5, 6]),  # [2,9] # 2
    np.array([0, 1, 2, 3, 5, 6]),  # [3,9] # 3
    np.array([0, 1, 2, 3, 5, 6]),  # [4,7] # 4
    np.array([0, 1, 2, 3, 4, 5, 6]),  # [6,5] # 5
    np.array([0, 1, 2, 3, 5, 6]),  # [8,3] # 6
    np.array([0, 1, 2, 3, 5, 6])  # [9,7] # 7
]


def main():
    with open('all_results_break10_realBSS_BREAK200MS_GCC.csv', 'w') as results:
        results.write(
            'Conf0,Conf1,Type,SIR1,SIR2,SDR1,SDR2,SAR1,SAR2\n')
        # for pos in [[10, 1]]:
        for I, pos in enumerate([[10, 6], [1, 8], [2, 9], [3, 9], [4, 7], [6, 5], [8, 3]]):
            # for pos in [[8, 3]]:
            #     for sample in ['A', 'B', 'C', 'D']:
            for sample in ['A', 'C']:
                print "#" * 10,
                print " {} - {} ".format(pos, sample),
                print "#" * 10
                base_dir = '/Data/Dropbox/Shared ISCAS2017Submissions/RecordingsDungeon/'
                [sampleRate, x_mix] = wavfile.read(
                    base_dir + 'dungeon_concurrent_{}_{}_{}_1.wav'.format(pos[0], pos[1], sample))

                TR = 10000 if sample in ['A', 'C'] and pos == [9, 7] else 18000
                trigger_index = np.where(x_mix > TR)[0][0]
                x_mix = x_mix[trigger_index + int(sampleRate / 32):].astype('float32')

                # Preprocessing params
                windowSize = 2048
                hopSize = 128
                windowFunction = hanning

                # TDOA params
                numTDOAs = 128

                # Input params
                microphoneSeparationInMetres = 0.5
                numSources = 3

                lp = 11000
                stereo_signal = np.array([butter_lowpass_filter(x_mix[:, 0], lp, sampleRate),
                                          butter_lowpass_filter(x_mix[:, 1], lp, sampleRate)]).T
                complexMixtureSpectrogram = computeComplexMixtureSpectrogram(stereo_signal.T, windowSize, hopSize,
                                                                             windowFunction)
                numChannels, numFrequencies, numTime = complexMixtureSpectrogram.shape
                frequenciesInHz = getFrequenciesInHz(sampleRate, numFrequencies)

                spectralCoherenceV = complexMixtureSpectrogram[0] * complexMixtureSpectrogram[1].conj() \
                                     / abs(complexMixtureSpectrogram[0]) / abs(complexMixtureSpectrogram[1])
                angularSpectrogram = getAngularSpectrogram(spectralCoherenceV, frequenciesInHz,
                                                           microphoneSeparationInMetres, numTDOAs)
                meanAngularSpectrum = mean(angularSpectrogram, axis=-1)
                targetTDOAIndexes = estimateTargetTDOAIndexesFromAngularSpectrum(meanAngularSpectrum,
                                                                                 microphoneSeparationInMetres,
                                                                                 numTDOAs, numSources)

                TH = 0.95
                LP = 5
                BW_ms = 200
                BW = int(np.ceil(BW_ms / 1000.0 * sampleRate / hopSize))

                B = low_pass(angularSpectrogram[targetTDOAIndexes[0], :], ll=LP)
                A = low_pass(angularSpectrogram[targetTDOAIndexes[-1], :], ll=LP)
                A -= np.min(A)
                B -= np.min(B)
                A /= np.max(A)
                B /= np.max(B)

                vad1 = np.float32(A * TH > B)
                vad2 = np.float32(B * TH > A)

                prefix_filenames = '/Data/Dropbox/spike_separation_joy/recordings_dungeon/dungeon_concurrent_'

                filename_id = '%d_%d_%s' % (pos[0], pos[1], sample)
                filename_whisper1 = prefix_filenames + filename_id + '_1.wav'
                filename_whisper3 = prefix_filenames + filename_id + '_3.wav'
                filename_groundtruth1 = '/Data/Dropbox/spike_separation_joy/recordings_dungeon/edited%s1_T.wav' % sample
                filename_groundtruth8 = '/Data/Dropbox/spike_separation_joy/recordings_dungeon/edited%s8_T.wav' % sample
                fs, whisper1 = wavfile.read(filename_whisper1)
                fs, whisper3 = wavfile.read(filename_whisper3)
                # plt.plot(whisper1[:115000, 0])
                # ipd.display(ipd.Audio(whisper1[:200000, 0], rate=fs))
                TR = 10000 if sample in ['A', 'C'] and pos == [9, 7] else 18000
                trigger_index = np.where(whisper1 > TR)[0][0]
                # print trigger_index
                # trigger_index = 115000
                whisper1 = whisper1[trigger_index + int(fs / 32):].astype('float32')
                whisper3 = whisper3[trigger_index + int(fs / 32):].astype('float32')

                # make the same length
                shortest_size = np.amin((whisper1.shape[0], whisper3.shape[0]))
                whisper1_shortened = whisper1[:shortest_size, :]
                whisper3_shortened = whisper3[:shortest_size, :]
                whisper = np.concatenate((whisper1_shortened, whisper3_shortened), 1)

                # gt
                fs, groundtruth1 = wavfile.read(filename_groundtruth1)
                groundtruth1 = remove_trigger(groundtruth1)
                fs, groundtruth8 = wavfile.read(filename_groundtruth8)
                groundtruth8 = remove_trigger(groundtruth8)

                # stfts & VADs
                m2 = stft(groundtruth8, windowSize, hopSize, window=np.hanning).T
                m1 = stft(groundtruth1, windowSize, hopSize, window=np.hanning).T
                m1_sum = np.sum(np.log(m1 + 1), axis=-1)
                m2_sum = np.sum(np.log(m2 + 1), axis=-1)
                VAD1 = np.array(0.8 * m1_sum > m2_sum)

                VAD2 = np.array(0.8 * m2_sum > m1_sum)

                x_spec = np.transpose(multi_stft(whisper.T, windowSize, hopSize), (0, 2, 1))

                vad1 = vad1[:len(VAD1)]
                vad2 = vad2[:len(VAD2)]
                JIT = 2

                ch0 = np.array([0, 1, 2, 5, 6, 3])
                ch1 = np.array([0, 1, 2, 5, 6, 3])

                _, _, vad_slice = triggers(sample, sampleRate, hopSize, JIT)
                _, _, spec_slice = triggers(sample, sampleRate, hopSize)
                x_spec = x_spec[:, :len(vad1), :]

                recons = []
                for vad, ch in zip([vad1, vad2], [ch0, ch1]):
                    # vad = vad[vad_slice]
                    vad = np.expand_dims(np.expand_dims(break_small(vad, BW), 0), -1)
                    # vad = np.concatenate([vad[:, vad_slice[0]], vad[:, vad_slice[1]]], axis=1)
                    xx = []
                    for v_sl, x_sl in zip(vad_slice, spec_slice):
                        xx += [x_spec[ch, x_sl] * vad[:, v_sl]]
                    # x_spec_a = x_spec[:, spec_slice]
                    # x_spec_a = x_spec_a[:, :vad.shape[1]]
                    ll1 = estimate_fw_mapping(np.concatenate(xx, axis=1))
                    x_spec_b = x_spec[ch].T
                    w = step_bf(x_spec_b, ll1)
                    recons.append(np.einsum('ab,acb->ca', np.conj(w), x_spec_b))

                recons[0] = istft(recons[0].T, hopSize, window=np.hanning)
                recons[1] = istft(recons[1].T, hopSize, window=np.hanning)

                # correct for filter length
                init1 = 0
                init0 = int(26 * sampleRate) if sample == 'A' else int(13 * sampleRate)
                init2 = int(40 * sampleRate) if sample == 'A' else int(26 * sampleRate)
                SAR = []
                SIR = []
                SDR = []
                for WH in [1, 3]:
                    for CH in [0, 1, 2, 3]:
                        # other
                        s2 = 'B' if sample == 'A' else 'A'
                        TR = 10000 if s2 in ['A', 'C'] and pos == [9, 7] else 18000
                        filename_id = '%d_%d_%s' % (pos[0], pos[1], s2)
                        filename_whisper1 = prefix_filenames + filename_id + '_{}.wav'.format(WH)
                        fs, whisper1 = wavfile.read(filename_whisper1)
                        trigger_index = np.where(whisper1 > TR)[0][0]
                        whisperB = whisper1[trigger_index + int(fs / 32):, CH].astype('float32')

                        s3 = 'D' if sample == 'A' else 'B'
                        TR = 10000 if s3 in ['A', 'C'] and pos == [9, 7] else 18000
                        filename_id = '%d_%d_%s' % (pos[0], pos[1], s3)
                        filename_whisper1 = prefix_filenames + filename_id + '_{}.wav'.format(WH)
                        fs, whisper1 = wavfile.read(filename_whisper1)
                        trigger_index = np.where(whisper1 > TR)[0][0]
                        whisperC = whisper1[trigger_index + int(fs / 32):, CH].astype('float32')

                        whisperB = whisperB[init0:init2]
                        whisperC = whisperC[init0:init2]

                        #         ipd.display(ipd.Audio(whisperB, rate=fs))
                        #         ipd.display(ipd.Audio(whisperC, rate=fs))

                        recons0 = recons[0][init0:init2]
                        recons1 = recons[1][init0:init2]

                        whisperCC = whisperC
                        whisperBB = whisperB
                        end = 100000

                        pp = np.correlate(recons1[init1:init1 + end], whisperB[init1:init1 + end], 'full')
                        j1 = np.argmax(pp) - end

                        pp = np.correlate(recons0[init1:init1 + end], whisperC[init1:init1 + end], 'full')
                        j0 = np.argmax(pp) - end

                        #         print "{} || {}".format(j0, j1)

                        if j0 > 0:
                            recons0 = recons0[j0:]
                        else:
                            whisperCC = whisperCC[abs(j0):]

                        if j1 > 0:
                            recons1 = recons1[j1:]
                        else:
                            whisperBB = whisperBB[abs(j1):]

                        min_len = np.min([len(recons0), len(recons1), len(whisperBB), len(whisperCC)])
                        recons0 = recons0[:min_len]
                        recons1 = recons1[:min_len]
                        whisperCC = whisperCC[:min_len]
                        whisperBB = whisperBB[:min_len]

                        wavfile.write(
                            '/Data/software/PEASS-Software-v2.0.1/example/dPEASS_{}_{}_1.wav'.format(pos, sample),
                            fs,
                            recons0 / np.max(np.abs(recons0)))
                        wavfile.write(
                            '/Data/software/PEASS-Software-v2.0.1/example/dPEASS_{}_{}_2.wav'.format(pos, sample),
                            fs,
                            recons1 / np.max(np.abs(recons1)))
                        wavfile.write(
                            '/Data/software/PEASS-Software-v2.0.1/example/dPEASS_{}_{}_1_gt.wav'.format(pos,
                                                                                                        sample), fs,
                            whisperCC / np.max(np.abs(whisperCC)))
                        wavfile.write(
                            '/Data/software/PEASS-Software-v2.0.1/example/dPEASS_{}_{}_2_gt.wav'.format(pos,
                                                                                                        sample), fs,
                            whisperBB / np.max(np.abs(whisperBB)))

                        sdr, sir, sar, perm = mir_eval.separation.bss_eval_sources(np.array([whisperCC, whisperBB]),
                                                                                   np.array([recons0, recons1]))

                        SDR.append(sdr)
                        SAR.append(sar)
                        SIR.append(sir)

                sir = np.max(np.array(SIR), axis=0)
                sar = np.max(np.array(SAR), axis=0)
                sdr = np.max(np.array(SDR), axis=0)
                print "*" * 40
                print "SIR: {:.3} dB || {:.3} dB".format(sir[0], sir[1])
                print "SAR: {:.3} dB || {:.3} dB".format(sar[0], sar[1])
                print "SDR: {:.3} dB || {:.3} dB".format(sdr[0], sdr[1])

                results.write(
                    '{},{},{},{},{},{},{},{},{} \n'.format(pos[0], pos[1], sample, sir[0], sir[1],

                                                           sdr[0], sdr[1], sar[0], sar[1]))
                print


if __name__ == "__main__":
    main()
