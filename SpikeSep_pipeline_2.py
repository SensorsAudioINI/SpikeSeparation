from __future__ import division

import traceback
import warnings
import datetime
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
    np.array([0, 1, 2, 5, 6]),  # [3,9] # 4
    np.array([0, 1, 2, 3, 5, 6]),  # [4,7] # 5
    np.array([0, 1, 2, 3, 5, 6]),  # [6,5] # 6
    np.array([0, 1, 2, 3, 5, 6]),  # [8,3] # 7
    np.array([0, 1, 2, 3, 5, 6])  # [9,7] # 8
]

TYPE = "GCC"  # "MNICA"  "SO"

if TYPE == 'SO':
    # SO
    good_delays = [8, 6, 16, 12, 12, 18, 6, 16, 12, 14, 12, 18, 10, 0, 20, 14, 20, 12]
    # good_delays = [8, 6, 16, 12, 4, 12, 12, 6, 6, 4, 6, 8, 8, 2, 2, 12, 8, 12]
    BREAK = [200, 100, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 100, 100, 200, 100, 200, 200]
    TH = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.6, 0.7, 0.9, 0.6, 0.8, 0.7, 0.9]
    LP = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]

if TYPE == 'MNICA':
    # MNICA
    good_delays = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    BREAK = [200, 100, 100, 100, 200, 200, 200, 100, 100, 200, 200, 200, 200, 200, 200, 200, 200, 200, ]
    TH = [0.5, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.7, 0.6]
    LP = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]

if TYPE == 'GCC':
    # GCC
    good_delays = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    BREAK = [200, 200, 200, 200, 200, 200, 200, 200, 100, 200, 200, 200, 200, 200, 200, 200, 200, 200]
    TH = [0.5, 0.5, 0.5, 0.5, 0.6, 0.5, 0.5, 0.7, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    LP = [10, 10, 10, 10, 10, 20, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 20]


sides = [[0, -1], [0, -1], [0, -1], [0, -1], [0, -1], [0, -1], [1, -1], [0, -1], [0, 1]]
TIME = datetime.datetime.now().isoformat().replace(':', '').replace('-', '')[:15]


def main():
    with open('frames_test__{}__{}.csv'.format(TIME, TYPE), 'w') as results:
        results.write(
            'Conf0,Conf1,Type,EstPos1,EstPos2,GoodMics,'
            'SIR1,GTSIR1,BESTSIR1,SIR2,GTSIR2,BESTSIR2,'
            'SDR1,GTSDR1,BESTSDR1,SDR2,GTSDR2,BESTSDR2,'
            'SAR1,GTSAR1,BESTSAR1,SAR2,GTSAR2,BESTSAR2,'
            'CO1,CO2,SNR,JIT,BREAK,TH,LP,PRE,RECALL,TOTFRA\n')
        for I, pos in enumerate([[10, 1], [10, 6], [1, 8], [2, 9], [3, 9], [4, 7], [6, 5], [8, 3], [9, 7]]):
            # for I, pos in enumerate([[6, 5], [9, 7]]):
            # for I, pos in enumerate([[6, 5]]):
            for J, sample in enumerate(['A', 'C']):
                # for th in [0.5, 0.6, 0.7, 0.8, 0.9]:
                #     for b in [100, 200]:
                #         for jit in [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:
                # for lp in [10, 20]:
                # b = 200
                # th = 0.9
                jit = good_delays[I * 2 + J]
                b = BREAK[I * 2 + J]
                # b = BR
                S = sides[I]
                th = TH[I * 2 + J]
                lp = LP[I * 2 + J]

                sad = SAD(pos, sample, frame_len=2048, frame_step=512, break_w_ms=b,
                          jit=jit)  # frame_step=120 for MNICA

                if TYPE == 'SO':
                    sad = SAD(pos, sample, frame_len=2048, frame_step=512, break_w_ms=b,
                              jit=jit)  # frame_step=120 for MNICA
                    sad_result = sad.get_SOSAD(frame_len=8192, frame_step=2048, th=th)

                if TYPE == 'GCC':
                    sad = SAD(pos, sample, frame_len=2048, frame_step=512, break_w_ms=b,
                              jit=jit)  # frame_step=120 for MNICA
                    sad_result = sad.get_GCCSAD(frame_len=1024, frame_step=128, th=th, lp=lp, sides=S)

                if TYPE == 'MNICA':
                    sad = SAD(pos, sample, frame_len=2048, frame_step=120, break_w_ms=b,
                              jit=jit)  # frame_step=120 for MNICA
                    sad_result = sad.get_MNICASAD(cut_f=20, th=th, res_f=200)

                if TYPE == "BESTMIC":
                    sad = SAD(pos, sample, frame_len=2048, frame_step=512, break_w_ms=b,
                              jit=jit)  # frame_step=120 for MNICA
                    recons = sad.best_mic(CHS[I])
                    sdr_sad, sir_sad, sar_sad, _ = sad.evaluate_bss(recons,
                                    save_name="{}sad".format(TYPE))
                else:
                    sad1, sad2 = sad_result['sad1'], sad_result['sad2']

                    sdr_sad, sir_sad, sar_sad, pre, recall, frames = sad.evaluate_bss(sad.apply_sad(sad1, sad2, CHS[I]),)
                                                                    # save_name="{}sad".format(TYPE))

                sdr_gt, sir_gt, sar_gt, _, _, _ = sad.evaluate_bss(sad.emp_limit_LCMV(CHS[I]))

                sir_best, sdr_best, sar_best = calculate_best(sample, pos, use_cached=True)

                # visualize
                headers = ["{} - {}".format(pos, sample), "SOSAD1", "W/ GT1", "BEST1", "EDIT", "SOSAD2", "W/ GT2",
                           "BEST2"]
                table = [["SIR", sir_sad[0], sir_gt[0], sir_best[0], pre, sir_sad[1], sir_gt[1], sir_best[1]],
                         ["SAR", sar_sad[0], sar_gt[0], sar_best[0], recall, sar_sad[1], sar_gt[1], sar_best[1]],
                         ["SDR", sdr_sad[0], sdr_gt[0], sdr_best[0], np.mean(frames), sdr_sad[1], sdr_gt[1], sdr_best[1]]]

                print
                print tabulate(table, headers, tablefmt="fancy_grid")

                results.write(
                    '{},{},{},{},{},{},{},{},'
                    '{},{},{},{},{},{},{},{},'
                    '{},{},{},{},{},{},{},{},'
                    '{},{},{},{},{},{},{},{},{},{} \n'.format(pos[0], pos[1], sample, 0,
                                                     0, CHS[I],
                                                     sir_sad[0], sir_gt[0], sir_best[0], sir_sad[1], sir_gt[1],
                                                     sir_best[1],
                                                     sdr_sad[0], sdr_gt[0], sdr_best[0], sdr_sad[1], sdr_gt[1],
                                                     sdr_best[1],
                                                     sar_sad[0], sar_gt[0], sar_best[0], sar_sad[1], sar_gt[1],
                                                     sar_best[1], 0, 0, sad.SNR, jit, b, th, lp, pre, recall, np.mean(frames)))


if __name__ == "__main__":
    # sir_b, sdr_b, sar_b = calculate_best('C', [6, 5], use_cached=False)
    main()

