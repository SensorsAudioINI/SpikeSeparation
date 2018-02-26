from __future__ import division

import numpy as np
import time
import sys

import progressbar


def get_itds(timestamps, ears, types, max_itd=800e-6, save_to_file=None, verbose=False, return_itd_indices=False):
    """Get the itds based on the event information, based on a basic deterministic algorithm.

    For every event e_i, the algorithm looks at the nearest event in time from the corresponding ear with every other
    attribute being the same (like the neuron id, channel, filterbank id, on_off id). The difference in time between the
    event e_i and the nearest event is the itd for the event e_i. Note that itd should be within max_itd, which leads
    to many events not having a valid itd.

    Args:
        :param timestamps: The timestamps, in np.float32 format.
        :param ears: The ear ids, as a numpy array. Expected in integer format, but not a strict restriction.
        :param types: The type ids for the events. The type id is a combined attribute, taking into account every
        attribute of an event except the ear attribute.
        :param max_itd: The maximum possible itd allowed for an event, in seconds. Defaults to 800us.
        :param save_to_file: If not None, a filename is expected, to which the itds and other attributes are saved.
        :param verbose: If True, displays a progressbar of the progress of the function.
        :param return_itd_indices: If True, the function returns the itd_indices array, matching every itd value to the
        event index in the original event stream.

    Returns:
        :return: A tuple (itds_to_return, itd_indices).
        itds_to_return: A numpy array, of dtype np.float32, with the itds. Note that the number of itds in the array
                        does not correspond to the size of the arrays timestamps, ears, types. This is because not all
                        events have a valid itd within the max_itd parameter.
        itd_indices: A numpy array, matching every itd to the event index in the original event stream.
    """
    ears = ears.astype(np.bool)
    itds_to_return = np.zeros(timestamps.size, dtype=np.float32)
    itds_to_return.fill(-5. * max_itd)

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
        timestamps_left = np.array(timestamps_dict[True][type_of_event])
        timestamp_indices_left = timestamp_indices_dict[True][type_of_event]
        timestamps_right = np.array(timestamps_dict[False][type_of_event])
        timestamp_indices_right = timestamp_indices_dict[False][type_of_event]

        for ts_right, ts_idx_right in zip(timestamps_right, timestamp_indices_right):
            matched_indices = np.where((timestamps_left >= ts_right - max_itd) &
                                       (timestamps_left < ts_right + max_itd))[0]
            if matched_indices.size > 0:
                matched_itds = ts_right - timestamps_left[matched_indices]
                min_itd = np.argmin(np.abs(matched_itds))
                itds_to_return[ts_idx_right] = matched_itds[min_itd]

        for ts_left, ts_idx_left in zip(timestamps_left, timestamp_indices_left):
            matched_indices = np.where((timestamps_right >= ts_left - max_itd) &
                                       (timestamps_right < ts_left + max_itd))[0]
            if matched_indices.size > 0:
                matched_itds = timestamps_right[matched_indices] - ts_left
                min_itd = np.argmin(np.abs(matched_itds))
                itds_to_return[ts_idx_left] = matched_itds[min_itd]

    itd_indices = np.where(itds_to_return > -4. * max_itd)[0]
    itds_to_return = itds_to_return[itd_indices]
    if save_to_file is not None:
        np.savez(save_to_file, timestamps=timestamps[itd_indices], ears=ears[itd_indices], types=types[itd_indices],
                 itds=itds_to_return, itd_indices=itd_indices)

    if return_itd_indices:
        return itds_to_return, itd_indices

    return itds_to_return


def get_itd_dict(max_itd, num_bins):
    """Gets the mapping between the quantized itds and their indices in array format.

    For example, if max_itd was 10 and number of bins are 4, then the mapping is [-7.5, -2.5, 2.5, 7.5] so that the itds
    in [-10, -5] get mapped to -7.5, itds in [-5, 0] get mapped to -2.5 and so on.

    Args:
        :param max_itd: The max_itd parameter for the algorithm, in seconds.
        :param num_bins: The num of bins to discretize the itds into, an integer.

    Returns:
        :return: A numpy array, of type float32.
    """
    bin_length = 2 * max_itd / num_bins
    return np.array([-max_itd + (idx + 0.5) * bin_length for idx in range(num_bins)], dtype=np.float32)


def get_itds_v2(timestamps, ears, types, max_itd=800e-6, save_to_file=None, verbose=False, return_attributes=False):
    """Get the itds based on the event information, based on a basic stochastic algorithm.

    For every event e_i, the algorithm looks at all the events within max_itd on either side from the corresponding ear
    with every other attribute being the same (like the neuron id, channel, filterbank id, on_off id). If there are no
    such close by events with the criteria, then the event e_i has no itd. If there are events with the criteria, then
    a random event is picked, and the difference in time between the event e_i and this random event is the itd value
    assigned to the event e_i.

    Args:
        :param timestamps: The timestamps, in np.float32 format.
        :param ears: The ear ids, as a numpy array. Expected in integer format, but not a strict restriction.
        :param types: The type ids for the events. The type id is a combined attribute, taking into account every
        attribute of an event except the ear attribute.
        :param max_itd: The maximum possible itd allowed for an event, in seconds. Defaults to 800us.
        :param save_to_file: If not None, a filename is expected, to which the itds and other attributes are saved.
        :param verbose: If True, displays a progressbar of the progress of the function.
        :param return_attributes: If True, the attributes of the events for which the itds are generated are also
        returned.

    Returns:
        :return: Either itds or a tuple (itds, timestamps, ears, types)
    """
    ears = ears.astype(np.bool)
    itds_to_return, timestamps_to_return, ears_to_return, types_to_return = [], [], [], []

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
        timestamps_left = np.array(timestamps_dict[True][type_of_event])
        timestamp_indices_left = timestamp_indices_dict[True][type_of_event]
        timestamps_right = np.array(timestamps_dict[False][type_of_event])
        timestamp_indices_right = timestamp_indices_dict[False][type_of_event]

        for ts_right, ts_idx_right in zip(timestamps_right, timestamp_indices_right):
            matched_indices = np.where((timestamps_left >= ts_right - max_itd) &
                                       (timestamps_left < ts_right + max_itd))[0]
            for matched_index in matched_indices:
                matched_itd = ts_right - timestamps_left[matched_index]
                itds_to_return.append(matched_itd)
                timestamps_to_return.append(ts_right)
                ears_to_return.append(False)
                types_to_return.append(type_of_event)

        for ts_left, ts_idx_left in zip(timestamps_left, timestamp_indices_left):
            matched_indices = np.where((timestamps_right >= ts_left - max_itd) &
                                       (timestamps_right < ts_left + max_itd))[0]
            for matched_index in matched_indices:
                matched_itd = timestamps_right[matched_index] - ts_left
                itds_to_return.append(matched_itd)
                timestamps_to_return.append(ts_left)
                ears_to_return.append(True)
                types_to_return.append(type_of_event)

    indices = np.argsort(timestamps_to_return)
    timestamps_to_return = np.array(timestamps_to_return, dtype=np.float32)[indices]
    itds_to_return = np.array(itds_to_return, dtype=np.float32)[indices]
    types_to_return = np.array(types_to_return, dtype=np.int16)[indices]
    ears_to_return = np.array(ears_to_return, dtype=np.int8)[indices]

    if save_to_file is not None:
        np.savez(save_to_file, timestamps=timestamps_to_return, ears=ears_to_return,
                 types=types_to_return, itds=itds_to_return)

    if return_attributes:
        return itds_to_return, timestamps_to_return, ears_to_return, types_to_return

    return itds_to_return


def get_itds_v3(timestamps, ears, types, max_itd=800e-6, save_to_file=None, verbose=False, return_itd_indices=False):
    """Get the itds based on the event information, based on a basic deterministic algorithm.

    This function implements the version 3 of the itd algorithm, which is basically the version 1 (get_itds) but with a condition.
    For event e_i, the itd pair event itd_e_i is the nearest event in time from the corresponding ear with every attribute being the same.
    The additional condition compared to version 1 is that the itd pair event for the event itd_e_i should be e_i.

    Args:
        :param timestamps: The timestamps, in np.float32 format.
        :param ears: The ear ids, as a numpy array. Expected in integer format, but not a strict restriction.
        :param types: The type ids for the events. The type id is a combined attribute, taking into account every
        attribute of an event except the ear attribute.
        :param max_itd: The maximum possible itd allowed for an event, in seconds. Defaults to 800us.
        :param save_to_file: If not None, a filename is expected, to which the itds and other attributes are saved.
        :param verbose: If True, displays a progressbar of the progress of the function.
        :param return_itd_indices: If True, the function returns the itd_indices array, matching every itd value to the
        event index in the original event stream.

    Returns:
        :return: A tuple (itds_to_return, itd_indices).
        itds_to_return: A numpy array, of dtype np.float32, with the itds. Note that the number of itds in the array
                        does not correspond to the size of the arrays timestamps, ears, types. This is because not all
                        events have a valid itd within the max_itd parameter.
        itd_indices: A numpy array, matching every itd to the event index in the original event stream.
    """
    ears = ears.astype(np.bool)
    itds_to_return = np.zeros(timestamps.size, dtype=np.float32)
    itds_to_return.fill(-5. * max_itd)

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

    max_num_events = 5

    for type_of_event in bar(np.unique(types)):
        timestamps_left = np.array(timestamps_dict[True][type_of_event])
        timestamp_indices_left = timestamp_indices_dict[True][type_of_event]
        timestamps_right = np.array(timestamps_dict[False][type_of_event])
        timestamp_indices_right = timestamp_indices_dict[False][type_of_event]

        num_right_events = timestamps_right.shape[0]

        for event_idx, (ts_right, ts_idx_right) in enumerate(zip(timestamps_right, timestamp_indices_right)):
            matched_indices = np.where((timestamps_left >= ts_right - max_itd) &
                                       (timestamps_left < ts_right + max_itd))[0]
            if matched_indices.size > 0:
                matched_itds = ts_right - timestamps_left[matched_indices]
                min_itd_idx_local = np.argmin(np.abs(matched_itds))
                min_itd = matched_itds[min_itd_idx_local]
                # absolute index of the itd pair event
                min_itd_ts_left = ts_right - min_itd
                # now check that the itd pair for the itd pair event is the current event
                if event_idx < max_num_events:
                    min_itd_ts_right = timestamps_right[0: event_idx + max_num_events + 1]
                    alt_min_itd_idx = np.argmin(np.abs(min_itd_ts_left - min_itd_ts_right))
                    if alt_min_itd_idx == event_idx:
                        itds_to_return[ts_idx_right] = min_itd
                else:
                    min_itd_ts_right = timestamps_right[event_idx - max_num_events: event_idx + max_num_events + 1]
                    alt_min_itd_idx = np.argmin(np.abs(min_itd_ts_left - min_itd_ts_right))
                    if alt_min_itd_idx == max_num_events:
                        itds_to_return[ts_idx_right] = min_itd
                if min_itd_ts_right[0] > min_itd_ts_left - max_itd or min_itd_ts_right[-1] < min_itd_ts_left + max_itd:
                    print('[WARNING] The max_num_events is not enough, please check.')
                    sys.stdout.flush()

    itd_indices = np.where(itds_to_return > -4. * max_itd)[0]
    itds_to_return = itds_to_return[itd_indices]
    if save_to_file is not None:
        np.savez(save_to_file, timestamps=timestamps[itd_indices], ears=ears[itd_indices], types=types[itd_indices],
                 itds=itds_to_return, itd_indices=itd_indices)

    if return_itd_indices:
        return itds_to_return, itd_indices

    return itds_to_return


def get_averaged_itds(itds, timestamps, channels, time_len=100e-6, event_limit=5, channel_low=10, channel_high=20, version='v1'):
    averaged_itds = []
    corresponding_timestamps = []
    current_time = timestamps[0]
    current_itd, current_count = 0, 0
    for itd_idx, itd_current in enumerate(itds):
        if channels[itd_idx] <= channel_low or channels[itd_idx] >= channel_high:
            continue
        if timestamps[itd_idx] < current_time + time_len:
            current_itd += itd_current
            current_count += 1
            if version == 'v2':
                if current_count > event_limit:
                    averaged_itds.append(current_itd / current_count)
                    corresponding_timestamps.append(current_time)
                    current_time = timestamps[itd_idx]
                    current_itd = itd_current
                    current_count = 1
                continue
        else:
            if version == 'v1':
                if current_count >= event_limit:
                    averaged_itds.append(current_itd / current_count)
                    corresponding_timestamps.append(current_time)
            current_time = timestamps[itd_idx]
            current_itd = itd_current
            current_count = 1
    averaged_itds = np.array(averaged_itds)
    corresponding_timestamps = np.array(corresponding_timestamps)
    return averaged_itds, corresponding_timestamps


if __name__ == '__main__':
    import es

    test_timestamps, test_addresses = es.loadaerdat('../data/man_clean.aedat')
    test_timestamps, test_ears, test_types = es.decode_ams1b(test_timestamps, test_addresses)

    start = time.time()
    test_itds_v1 = get_itds(test_timestamps, test_ears, test_types, save_to_file='man_clean', verbose=True)
    test_itds_v2 = get_itds_v2(test_timestamps, test_ears, test_types, save_to_file='man_clean', verbose=True)
    print('Computing the itds complete, took {} seconds'.format(time.time() - start))
