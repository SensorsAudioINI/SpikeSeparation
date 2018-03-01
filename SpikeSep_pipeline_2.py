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
    step_bf, remove_trigger, triggers, break_small, prefix_filenames, basedir, calculate_best, SAD

warnings.filterwarnings("ignore")

CHS = [
    np.array([0, 1, 2, 3, 5, 6]),  # [10,1] # 0
    np.array([0, 1, 2, 3, 5, 6]),  # [10,6] # 1
    np.array([0, 1, 2, 3, 5, 6]),  # [1,8] # 2
    np.array([0, 1, 2, 5, 6]),  # [2,9] # 3
    np.array([0, 1, 2, 3, 5, 6]),  # [3,9] # 4
    np.array([0, 1, 2, 3, 5, 6]),  # [4,7] # 5
    np.array([0, 1, 2, 3, 5, 6]),  # [6,5] # 6
    np.array([0, 1, 2, 3, 5, 6]),  # [8,3] # 7
    np.array([0, 1, 2, 3, 5, 6])  # [9,7] # 8
]


# good_delays = [8, 6, 16, 12, 4, 12, 12, 6, 6, 4, 6, 8, 8, 2, 2, 12, 8, 12]
good_delays = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
BREAK = [200, 100, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 100, 100, 200, 200, 100, 200]


def main():
    with open('test_slsl_JIT_full_details_MNICA.csv', 'w') as results:
        results.write(
            'Conf0,Conf1,Type,EstPos1,EstPos2,GoodMics,'
            'SIR1,GTSIR1,BESTSIR1,SIR2,GTSIR2,BESTSIR2,'
            'SDR1,GTSDR1,BESTSDR1,SDR2,GTSDR2,BESTSDR2,'
            'SAR1,GTSAR1,BESTSAR1,SAR2,GTSAR2,BESTSAR2,'
            'CO1,CO2,SNR,JIT,BREAK\n')
        for I, pos in enumerate([[10, 1], [10, 6], [1, 8], [2, 9], [3, 9], [4, 7], [6, 5], [8, 3], [9, 7]]):
            # for I, pos in enumerate([[9, 7]]):
            for J, sample in enumerate(['A', 'C']):
                jit = good_delays[I * 2 + J]
                b = BREAK[I * 2 + J]
                # for sample in ['B']:

                sad = SAD(pos, sample, frame_len=2048, frame_step=512, break_w_ms=b, jit=jit)

                # sad_result = sad.get_SOSAD(frame_len=8192, frame_step=2048, th=0.9)

                # sad_result = sad.get_GCCSAD(frame_len=1024, frame_step=128)

                sad_result = sad.get_MNICASAD(cut_f=20, th=0.9)

                sad1, sad2 = sad_result['sad1'], sad_result['sad2']

                sdr_sad, sir_sad, sar_sad, _ = sad.evaluate_bss(sad.apply_sad(sad1, sad2, CHS[I]), save_name='mnicasad')

                sdr_gt, sir_gt, sar_gt, _ = sad.evaluate_bss(sad.emp_limit_LCMV(CHS[I]))

                sir_best, sdr_best, sar_best = calculate_best(sample, pos, use_cached=True)

                # visualize
                headers = ["{} - {}".format(pos, sample), "SOSAD1", "W/ GT1", "BEST1", "", "SOSAD2", "W/ GT2",
                           "BEST2"]
                table = [["SIR", sir_sad[0], sir_gt[0], sir_best[0], '', sir_sad[1], sir_gt[1], sir_best[1]],
                         ["SAR", sar_sad[0], sar_gt[0], sar_best[0], '', sar_sad[1], sar_gt[1], sar_best[1]],
                         ["SDR", sdr_sad[0], sdr_gt[0], sdr_best[0], '', sdr_sad[1], sdr_gt[1], sdr_best[1]]]

                print tabulate(table, headers, tablefmt="fancy_grid")

                results.write(
                    '{},{},{},{},{},{},{},{},'
                    '{},{},{},{},{},{},{},{},'
                    '{},{},{},{},{},{},{},{},'
                    '{},{},{},{},{} \n'.format(pos[0], pos[1], sample, sad_result['first'],
                                                  sad_result['second'], CHS[I],
                                                  sir_sad[0], sir_gt[0], sir_best[0], sir_sad[1], sir_gt[1], sir_best[1],
                                                  sdr_sad[0], sdr_gt[0], sdr_best[0], sdr_sad[1], sdr_gt[1], sdr_best[1],
                                                  sar_sad[0], sar_gt[0], sar_best[0], sar_sad[1], sar_gt[1], sar_best[1],
                                                  sad_result['corr1'], sad_result['corr2'], sad.SNR,
                                                  jit,b))


if __name__ == "__main__":
    # sir_b, sdr_b, sar_b = calculate_best('C', [6, 5], use_cached=False)
    main()
