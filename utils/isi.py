from __future__ import division

import numpy as np
import time

import progressbar


def get_isis(timestamps, ears, types, save_to_file=None, verbose=False, return_isi_attributes=False):
    """Get the inter spike intervals based on the event information.

    Args:
        :param timestamps: The timestamps, in np.float32 format.
        :param ears: The ear ids, as a numpy array. Expected in integer format, but not a strict restriction.
        :param types: The type ids for the events. The type id is a combined attribute, taking into account every
        attribute of an event except the ear attribute.
        :param save_to_file: If not None, a filename is expected, to which the isis and other attributes are saved.
        :param verbose: If True, displays a progressbar of the progress of the function.
        :param return_isi_attributes: If True, the function returns the other attributes such as timestamps, ears, types

    Returns:
        :return: A numpy array, isis.
    """
    ears = ears.astype(np.bool)
    isis_to_return = np.zeros(timestamps.size, dtype=np.float32)
    isis_to_return.fill(-5.)

    timestamps_dict = {}
    timestamp_indices_dict = {}
    for ear in np.unique(ears):
        timestamps_dict[ear] = {}
        timestamp_indices_dict[ear] = {}
        for type_of_event in np.unique(types):
            timestamps_dict[ear][type_of_event] = []
            timestamp_indices_dict[ear][type_of_event] = []

    for idx, (timestamp, ear, type_of_event) in enumerate(zip(timestamps, ears, types)):
        timestamps_dict[ear][type_of_event].append(timestamp)
        timestamp_indices_dict[ear][type_of_event].append(idx)

    if verbose:
        print('Initialized the timestamp lists.')

    bar = progressbar.ProgressBar() if verbose else lambda x: x

    for type_of_event in bar(np.unique(types)):
        for ear in np.unique(ears):
            c_timestamps = np.array(timestamps_dict[ear][type_of_event])
            c_timestamp_indices = np.array(timestamp_indices_dict[ear][type_of_event])
            
            for ts, ts_p, ts_idx in zip(c_timestamps[1:], c_timestamps[:-1], c_timestamp_indices[1:]):
                isis_to_return[ts_idx] = ts - ts_p
    
    isi_indices = np.where(isis_to_return > -4.)[0]
    isis_to_return = isis_to_return[isi_indices]
    if save_to_file is not None:
        np.savez(save_to_file, timestamps=timestamps[isi_indices], ears=ears[isi_indices], types=types[isi_indices],
                 isis=isis_to_return, isi_indices=isi_indices)

    if return_isi_attributes:
        return isis_to_return, timestamps[isi_indices], ears[isi_indices], types[isi_indices]

    return isis_to_return