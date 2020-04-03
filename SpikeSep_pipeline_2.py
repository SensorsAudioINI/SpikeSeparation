from __future__ import division

import datetime
import warnings

import numpy as np
from tabulate import tabulate

from SpikeSep import calculate_best, SAD

import pandas as pd

warnings.filterwarnings("ignore")

# channel 4 and 7 are generally corrupted -> mic overflow

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

# for GCC sometimes the peaks are on the same side (with respect to the 0)
sides = [[0, -1], [0, -1], [0, -1], [0, -1], [0, -1], [0, -1], [1, -1], [0, -1], [0, 1]]

TIME = datetime.datetime.now().isoformat().replace(':', '').replace('-', '')[:15]

columns = ['Conf0', 'Conf1', 'Type', 'EstPos1', 'EstPos2', 'GoodMics', 'SIR1', 'GTSIR1', 'BESTSIR1',
           'SIR2', 'GTSIR2', 'BESTSIR2', 'SDR1', 'GTSDR1', 'BESTSDR1', 'SDR2', 'GTSDR2', 'BESTSDR2',
           'SAR1', 'GTSAR1', 'BESTSAR1', 'SAR2', 'GTSAR2', 'BESTSAR2', 'CO1', 'CO2', 'SNR', 'JIT',
           'BREAK', 'TH', 'LP', 'PRE', 'RECALL', 'TOTFRA']


def main():
    df = pd.DataFrame(columns=columns)
    # for I, pos in enumerate([[10, 1], [10, 6], [1, 8], [2, 9], [3, 9], [4, 7], [6, 5], [8, 3], [9, 7]]):
    for I, pos in enumerate([[4, 7], [6, 5], [8, 3], [9, 7]]):
        for J, sample in enumerate(['A', 'C']):
            jit = good_delays[I * 2 + J]
            b = BREAK[I * 2 + J]
            S = sides[I]
            th = TH[I * 2 + J]
            lp = LP[I * 2 + J]

            sad = SAD(pos, sample, frame_len=2048, frame_step=512, break_w_ms=b,
                      jit=jit)  # frame_step=120 for MNICA

            if TYPE == 'SO':
                sad = SAD(pos, sample, frame_len=2048, frame_step=512, break_w_ms=b,
                          jit=jit)
                sad_result = sad.get_SOSAD(frame_len=8192, frame_step=2048, th=th)

            if TYPE == 'GCC':
                sad = SAD(pos, sample, frame_len=2048, frame_step=512, break_w_ms=b,
                          jit=jit)
                sad_result = sad.get_GCCSAD(frame_len=1024, frame_step=128, th=th, lp=lp, sides=S)

            if TYPE == 'MNICA':
                sad = SAD(pos, sample, frame_len=2048, frame_step=120, break_w_ms=b,
                          jit=jit)
                sad_result = sad.get_MNICASAD(cut_f=20, th=th, res_f=200)

            if TYPE == "BESTMIC":
                sad = SAD(pos, sample, frame_len=2048, frame_step=512, break_w_ms=b,
                          jit=jit)
                recons = sad.best_mic(CHS[I])
                sdr_sad, sir_sad, sar_sad, _ = sad.evaluate_bss(recons,
                                                                save_name="{}sad".format(TYPE))
            else:
                sad1, sad2 = sad_result['sad1'], sad_result['sad2']

                sdr_sad, sir_sad, sar_sad, pre, recall, frames = sad.evaluate_bss(sad.apply_sad(sad1, sad2, CHS[I]),
                                                                                  save_name="{}sad".format(TYPE))

            sdr_gt, sir_gt, sar_gt, _, _, _ = sad.evaluate_bss(sad.emp_limit_LCMV(CHS[I]))

            sir_best, sdr_best, sar_best = calculate_best(sample, pos, use_cached=True)

            # visualize
            headers = ["{} - {}".format(pos, sample), "SOSAD1", "W/ GT1", "BEST1", "EDIT", "SOSAD2", "W/ GT2",
                       "BEST2"]
            table = [["SIR", sir_sad[0], sir_gt[0], sir_best[0], pre, sir_sad[1], sir_gt[1], sir_best[1]],
                     ["SAR", sar_sad[0], sar_gt[0], sar_best[0], recall, sar_sad[1], sar_gt[1], sar_best[1]],
                     ["SDR", sdr_sad[0], sdr_gt[0], sdr_best[0], np.mean(frames), sdr_sad[1], sdr_gt[1],
                      sdr_best[1]]]

            print
            print tabulate(table, headers, tablefmt="fancy_grid")

            new_df = pd.DataFrame([[pos[0], pos[1], sample, 0, 0, CHS[I], sir_sad[0], sir_gt[0], sir_best[0],
                                    sir_sad[1], sir_gt[1], sir_best[1], sdr_sad[0], sdr_gt[0], sdr_best[0],
                                    sdr_sad[1], sdr_gt[1], sdr_best[1], sar_sad[0], sar_gt[0], sar_best[0],
                                    sar_sad[1], sar_gt[1], sar_best[1], 0, 0, sad.SNR, jit, b, th, lp, pre,
                                    recall, np.mean(frames)]], columns=columns)

            df = df.append(new_df)
            # cache
            df.to_csv('frames_test__{}__{}.csv'.format(TIME, TYPE))


if __name__ == "__main__":
    main()
