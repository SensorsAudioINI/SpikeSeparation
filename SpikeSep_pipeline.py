from __future__ import division

import traceback
import warnings

import mir_eval
import numpy as np
from librosa import stft, istft
from scipy.io import wavfile
import matplotlib.pyplot as plt
from tabulate import tabulate

from SpikeSep import get_timestamps_from_filename, calculate_itds, assign_the_spikes, multi_stft, estimate_fw_mapping, \
    step_bf, remove_trigger, triggers, break_small, prefix_filenames, basedir, calculate_best

warnings.filterwarnings("ignore")


CHS = [
    np.array([0, 1, 2, 3, 5, 6]),  # [10,1] # 0
    np.array([0, 1, 2, 3, 5, 6]),  # [10,6] # 1
    np.array([0, 1, 2, 3, 5, 6]),  # [1,8] # 2
    np.array([0, 1, 2, 5, 6]),  # [2,9] # 3
    np.array([0, 1, 2, 3, 5, 6]),  # [3,9] # 4
    np.array([0, 1, 2, 3, 5, 6]),  # [4,7] # 5
    np.array([0, 1, 2, 3, 4, 5, 6]),  # [6,5] # 6
    np.array([0, 1, 2, 3, 5, 6]),  # [8,3] # 7
    np.array([0, 1, 2, 3, 5, 6])  # [9,7] # 8
]

TH = 0.9
break_w_ms = 100  # ms

I = 6


def main():
    with open('all_results_break10_realBSS_BREAK200MS_39.csv', 'w') as results:
        results.write(
            'Conf0,Conf1,Type,EstPos1,EstPos2,GoodMics,SIR1,best1,SIR2,best2,CO1,CO2,SDR1,SDR2,SAR1,SAR2,SNR\n')
        # for I, pos in enumerate([[10, 1], [10, 6], [1, 8], [2, 9], [3, 9], [4, 7], [6, 5], [8, 3], [9, 7]]):
        for pos in [[6, 5]]:
            for sample in ['A', 'C']:
                # for sample in ['B']:

                try:
                    # get all the filenames
                    filename_id = '{}_{}_{}'.format(pos[0], pos[1], sample)
                    filename_cochlea = prefix_filenames + filename_id
                    filename_whisper1 = prefix_filenames + filename_id + '_1.wav'
                    filename_whisper3 = prefix_filenames + filename_id + '_3.wav'
                    filename_groundtruth1 = basedir + '/edited%s1_T.wav' % sample
                    filename_groundtruth8 = basedir + '/edited%s8_T.wav' % sample

                    # compute estimates
                    timestamps, ear_id, type_id, channel_id = get_timestamps_from_filename(filename_cochlea)

                    timestamps, ear_id, type_id, channel_id, itds = calculate_itds(timestamps, ear_id, type_id,
                                                                                   channel_id,
                                                                                   return_itd_indices=True)

                    estimates, argmax_estimates = assign_the_spikes(itds, sigma=45)

                    # # params

                    # ground truth
                    fs, groundtruth1 = wavfile.read(filename_groundtruth1)
                    groundtruth1 = remove_trigger(groundtruth1)
                    fs, groundtruth8 = wavfile.read(filename_groundtruth8)
                    groundtruth8 = remove_trigger(groundtruth8)

                    # params
                    fs = 24000.0
                    frame_len = 8192
                    frame_len_s = frame_len / fs
                    frame_step = 2048
                    frame_step_s = frame_step / fs

                    TH1 = TH
                    TH2 = TH
                    # print "jitTER is {}".format(jit)

                    mag1 = stft(groundtruth1, frame_len, frame_step, window=np.hanning)
                    e1 = np.sum(np.log(np.abs(mag1) + 1), axis=0)
                    mag8 = stft(groundtruth8, frame_len, frame_step, window=np.hanning)
                    e8 = np.sum(np.log(np.abs(mag8) + 1), axis=0)
                    r_ch = range(10, 20)
                    # fig = plt.figure(figsize=(16,4))
                    scores = []
                    envs = []
                    for idx in range(estimates.shape[1]):
                        #     a = timestamps[argmax_estimates==idx & np.array([k in r_ch for k in channel_id])]
                        a = timestamps[argmax_estimates == idx]
                        f_ch = channel_id[argmax_estimates == idx]
                        summed_prob = np.sum(estimates[:, idx]) / len(timestamps)
                        if len(a) == 0:
                            envs.append(np.zeros((len(e1),)))
                            continue
                        a = a[np.array([k in r_ch for k in f_ch])]
                        ts = np.arange(0, frame_step_s * (len(e1) + 1), frame_step_s)
                        c = [len(a[(a > t) & (a < (t + frame_len_s))]) for i, t in enumerate(ts[:-1])]
                        #         spks = a[(a > (t - frame_step_s)) & (a < t )]
                        #         probs = a_prob[(a > (t - frame_step_s)) & (a < t)]
                        #         c.append(np.sum(spks * probs))

                        _score = summed_prob * len(a)
                        scores.append(_score)
                        envs.append(c)

                    corr_envs = np.corrcoef(envs)
                    weights = 0.005 ** (corr_envs[np.argmax(scores)] + 1)

                    first = np.argmax(scores) + 1
                    second = np.argmax(np.log(scores * weights)) + 1

                    if first != second:
                        pass
                        # print "Estimated [{}, {}] == True {}".format(first, second, pos)
                        # out.write("Estimated [{}, {}] == True {} + \n".format(first, second, pos))
                    else:
                        third = np.argsort(np.log(scores * weights))[-2] + 1
                        second = third
                        # print "Estimated [{}, {}] == True {}".format(first, third, pos)
                        # out.write("Estimated [{}, {}] == True {} + \n".format(first, third, pos))

                    estimated_env_1 = np.log(np.array(envs[first - 1]) + 1)
                    estimated_env_2 = np.log(np.array(envs[second - 1]) + 1)

                    # print "SCORES {} || {}".format(scores[first - 1], scores[second - 1])

                    corr_env_1 = np.corrcoef(estimated_env_1, e1)[0, 1]
                    corr_env_2 = np.corrcoef(estimated_env_2, e8)[0, 1]
                    # print "{} {}".format(corr_env_1, corr_env_2)

                    # estimated_env_1 = np.log(np.clip(np.array(ccs[first - 1]) - non_envs, 0.0, np.inf) + 1)
                    # estimated_env_2 = np.log(np.clip(np.array(ccs[second - 1])  - non_envs, 0.0, np.inf) + 1)

                    # estimated_env_1 = estimated_env_1 / np.max(estimated_env_1)
                    # estimated_env_2 = estimated_env_2 / np.max(estimated_env_2)

                    # E1 = np.sum(estimated_env_1 ** 2)
                    # E2 = np.sum(estimated_env_2 ** 2)
                    # p_ratio = E1 / E2
                    # print "E1 = {} - E8 = {}".format(E1, E2)
                    #
                    # print "Power ratio = {}".format(E1 / E2)

                    # vad1 = np.float32(
                    #     (estimated_env_1 - TH * estimated_env_1 > estimated_env_2) & (estimated_env_1 > TH))
                    # vad2 = np.float32(
                    #     (estimated_env_2 - TH * estimated_env_2 > estimated_env_1) & (estimated_env_2 > TH))

                    vad1 = np.float32(estimated_env_1 - TH1 * estimated_env_1 > estimated_env_2)
                    vad2 = np.float32(estimated_env_2 - TH2 * estimated_env_2 > estimated_env_1)

                    frame_len = 2048
                    frame_step = 512
                    jit = int(frame_len / frame_step / 2)
                    break_w = int(np.ceil(break_w_ms / 1000.0 * fs / frame_step))
                    # print "JITTER is {}".format(JIT)
                    # load WHISPER data
                    # remove the trigger
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
                    whisper = np.concatenate((whisper1_shortened, whisper3_shortened), 1)

                    # stfts & VADs
                    m2 = stft(groundtruth8, frame_len, frame_step, window=np.hanning).T
                    m1 = stft(groundtruth1, frame_len, frame_step, window=np.hanning).T
                    m1_sum = np.sum(np.log(m1 + 1), axis=-1)
                    m2_sum = np.sum(np.log(m2 + 1), axis=-1)
                    VAD1 = np.array(0.8 * m1_sum > m2_sum)
                    VAD2 = np.array(0.8 * m2_sum > m1_sum)

                    x_spec = np.transpose(multi_stft(whisper.T, frame_len, frame_step, window=np.hanning), (0, 2, 1))

                    # solid_vad1 = np.tile(np.expand_dims(vad1, -1), (1, int(frame_len / 2 + 1)))
                    # solid_vad2 = np.tile(np.expand_dims(vad2, -1), (1, int(frame_len / 2 + 1)))
                    # mc1 = np.argmax(np.correlate(vad1, VAD1, "full")) - len(VAD1)
                    # mc2 = np.argmax(np.correlate(vad2, VAD2, "full")) - len(VAD2)
                    # print "Estimated JIT {} || Theoretical JIT = {}".format(np.mean([mc1, mc2]), jit)
                    # print "CORR: {:.4} || {:.4} ==> {:.4} || {:.4}".format(np.corrcoef(vad1, VAD1)[0, 1],
                    #                                                        np.corrcoef(vad2, VAD2)[0, 1],
                    #                                                        np.corrcoef(vad1[jit:], VAD1[:-jit])[0, 1],
                    #                                                        np.corrcoef(vad2[jit:], VAD2[:-jit])[0, 1])

                    slice_spk1, slice_spk2, _ = triggers(sample, fs, 1)

                    spk1 = []
                    spk2 = []
                    for i in range(8):
                        spk1 += [10 * np.log10(np.sum(whisper[slice_spk1, i] ** 2))]
                        spk2 += [10 * np.log10(np.sum(whisper[slice_spk2, i] ** 2))]

                    # print "PWR1 {} || PWR 2 {}".format(np.mean(spk1), np.mean(spk2))
                    SNR = np.mean(spk1) - np.mean(spk2)

                    R = int(np.floor(x_spec.shape[1] / len(vad1)))
                    vad1 = np.repeat(vad1, R)
                    vad2 = np.repeat(vad2, R)
                    VAD1 = VAD1[:min([len(vad1), len(VAD1)])]
                    VAD2 = VAD2[:min([len(vad2), len(VAD2)])]
                    vad1 = vad1[:min([len(vad1), len(VAD1)])]
                    vad2 = vad2[:min([len(vad2), len(VAD2)])]

                    CORR_AFTER1 = np.corrcoef(vad1[jit:], VAD1[:-jit])[0, 1]
                    CORR_AFTER2 = np.corrcoef(vad2[jit:], VAD2[:-jit])[0, 1]

                    # whisper_std = np.std(whisper - np.mean(whisper, axis=0, keepdims=True), axis=0)
                    # m_d = np.mean(whisper_std)
                    # s_d = np.std(whisper_std)
                    # z_score = (whisper_std - m_d) / s_d
                    # clean_ch = np.where(z_score < 0.9)[0]
                    # print z_score
                    # print "Good Channels: {}".format(clean_ch)

                    ch1 = ch2 = CHS[I]
                    # vad_slice = slice(int(fs * 26 / frame_step) + jit, int(fs * 53 / frame_step) + jit)
                    # spec_slice = slice(int(fs * 26 / frame_step), int(fs * 53 / frame_step))

                    gt_vad1, gt_vad2, vad_slice = triggers(sample, fs, frame_step, jit)
                    gt_s1, gt_s2, spec_slice = triggers(sample, fs, frame_step)
                    x_spec = x_spec[:, :len(vad1), :]

                    recons = []
                    for vad, val_ch in zip([vad1, vad2], [ch1, ch2]):
                        vad = np.expand_dims(np.expand_dims(break_small(vad, break_w), 0), -1)
                        xx = []
                        for v_sl, x_sl in zip(vad_slice, spec_slice):
                            xx += [x_spec[val_ch, x_sl] * vad[:, v_sl]]
                        ll1 = estimate_fw_mapping(np.concatenate(xx, axis=1))
                        x_spec_b = x_spec[val_ch].T
                        w = step_bf(x_spec_b, ll1)
                        recons.append(np.einsum('ab,acb->ca', np.conj(w), x_spec_b))

                    # EMP LIMIT
                    for vad, val_ch, x_sl, v_sl in zip([vad1, vad2], [ch1, ch2], [gt_vad1, gt_vad2], [gt_s1, gt_s2]):
                        xx = x_spec[val_ch, x_sl]
                        ll1 = estimate_fw_mapping(xx)
                        x_spec_b = x_spec[val_ch].T
                        w = step_bf(x_spec_b, ll1)
                        recons.append(np.einsum('ab,acb->ca', np.conj(w), x_spec_b))

                    # for i in range(len(recons)):
                    #     recons[0] = istft(recons[i].T, frame_step)
                    recons = [istft(x.T, frame_step, window=np.hanning) for x in recons]
                    # vad_slice = slice(int(fs * 30 / frame_step) + jit, int(fs * 50 / frame_step) + jit)
                    # spec_slice = slice(int(fs * 30 / frame_step), int(fs * 50 / frame_step))
                    #
                    # recons = []
                    # for vad in [vad1, vad2]:
                    #     x_speca = x_spec[clean_ch, :len(vad), :]
                    #     vad = np.expand_dims(np.expand_dims(vad, 0), -1)
                    #     LL1 = estimate_fw_mapping(x_speca[:, spec_slice] * vad[:, vad_slice])
                    #     x_specb = x_speca.T
                    #     W = step_bf(x_specb, LL1)
                    #     recons.append(np.einsum('ab,acb->ca', np.conj(W), x_specb))
                    #
                    # recons[0] = istft(recons[0].T, frame_step)
                    # recons[1] = istft(recons[1].T, frame_step)

                    # correct for filter length
                    init1 = 0
                    init0 = int(30 * fs) if sample == 'A' else int(16 * fs)
                    init2 = int(40 * fs) if sample == 'A' else int(26 * fs)
                    # init_1 = int(25 * fs)
                    recons0 = recons[0][init0:init2]
                    recons1 = recons[1][init0:init2]
                    groundtruth11 = groundtruth1[init0:]
                    groundtruth88 = groundtruth8[init0:]

                    end = 100000

                    pp = np.correlate(recons1[init1:init1 + end], groundtruth88[init1:init1 + end], 'full')
                    j1 = np.argmax(pp) - end

                    pp = np.correlate(recons0[init1:init1 + end], groundtruth11[init1:init1 + end], 'full')
                    j0 = np.argmax(pp) - end

                    # print "{} || {}".format(j0, j1)
                    # out.write("{} || {} + \n".format(j0, j1))

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

                    sdr, sir, sar, perm = mir_eval.separation.bss_eval_sources(np.array([groundtruth11, groundtruth88]),
                                                                               np.array([recons0, recons1]))
                    # print "SIR: {:.3} dB || {:.3} dB".format(sir[0], sir[1])

                    # wavfile.write('outputs/PEASS_{}_{}_1.wav'.format(pos, sample), fs,
                    #               recons0 / np.max(np.abs(recons0)))
                    # wavfile.write('outputs/PEASS_{}_{}_2.wav'.format(pos, sample), fs,
                    #               recons1 / np.max(np.abs(recons1)))
                    # wavfile.write('outputs/PEASS_{}_{}_1_gt.wav'.format(pos, sample), fs,
                    #               groundtruth11 / np.max(np.abs(groundtruth11)))
                    # wavfile.write('outputs/PEASS_{}_{}_2_gt.wav'.format(pos, sample), fs,
                    #               groundtruth88 / np.max(np.abs(groundtruth88)))

                    ## BEST
                    recons0 = recons[2][init0:init2]
                    recons1 = recons[3][init0:init2]
                    groundtruth11 = groundtruth1[init0:]
                    groundtruth88 = groundtruth8[init0:]

                    # wavfile.write(
                    #     '/Data/software/PEASS-Software-v2.0.1/example/PEASS_{}_{}_1_best.wav'.format(pos, sample),
                    #     fs,
                    #     recons0 / np.max(np.abs(recons0)))
                    # wavfile.write(
                    #     '/Data/software/PEASS-Software-v2.0.1/example/PEASS_{}_{}_2_best.wav'.format(pos, sample),
                    #     fs,
                    #     recons1 / np.max(np.abs(recons1)))

                    end = 100000

                    pp = np.correlate(recons1[:end], groundtruth88[:end], 'full')
                    j1 = np.argmax(pp) - end

                    pp = np.correlate(recons0[:end], groundtruth11[:end], 'full')
                    j0 = np.argmax(pp) - end

                    # print "{} || {}".format(j0, j1)
                    # out.write("{} || {} + \n".format(j0, j1))

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

                    sdr2, sir2, sar2, perm = mir_eval.separation.bss_eval_sources(
                        np.array([groundtruth11, groundtruth88]),
                        np.array([recons0, recons1]))

                    sir_b, sdr_b, sar_b = calculate_best(sample, pos, use_cached=True)

                    # visualize
                    headers = ["{} - {}".format(pos, sample), "SOSAD1", "W/ GT1", "BEST1", "", "SOSAD2", "W/ GT2", "BEST2"]
                    table = [["SIR", sir[0], sir2[0], sir_b[0], '', sir[1], sir2[1], sir_b[1]],
                             ["SAR", sar[0], sar2[0], sar_b[0], '', sar[1], sar2[1], sar_b[1]],
                             ["SDR", sdr[0], sdr2[0], sdr_b[0], '', sdr[1], sdr2[1], sdr_b[1]]]

                    print tabulate(table, headers, tablefmt="fancy_grid")

                except Exception, e:
                    out_name = 'error_{}_{}.txt'.format(pos, sample)
                    with open(out_name, 'w') as out:
                        print "Exit with exception {}".format(e)
                        out.write("##### + \n")
                        traceback.print_exc(file=out)
                        out.write("##### \n")
                    results.write(
                        '{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{} \n'.format(pos[0], pos[1], sample, -1, -1,
                                                                                       -1, -1, -1, -1, -1, -1, -1, -1,
                                                                                       -1, -1, -1, -1))
                    continue

                results.write(
                    '{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{} \n'.format(pos[0], pos[1], sample, first,
                                                                                   second, ch1, sir[0], sir2[0], sir[1],
                                                                                   sir2[1], CORR_AFTER1, CORR_AFTER2,
                                                                                   sdr[0], sdr[1], sar[0], sar[1], SNR))
                print


if __name__ == "__main__":
    main()
