from __future__ import print_function

import os
import warnings
try:
    from tkFileDialog import askopenfilename
except:
    from tkinter.filedialog import askopenfilename
import numpy as np

NB_CHANNELS = 64
NB_NEURON_TYPES = 4
NB_EARS = 2
NB_FILTER_BANKS = 2
NB_ON_OFF = 2


# noinspection PyTypeChecker
def loadaerdat(filename=None, curr_directory=False, max_events=30000000):
    """Gets the event timestamps and the corresponding addresses for a .aedat or a .dat file.

    The function implements in python the matlab function loadaerdat.m;
    this function was written by Zhe He (zhhe@ini.uzh.ch).

    Args:
        :param filename: (optional) The path to the .aedat or .dat file;
        the path is relative to the current directory if curr_directory is True, else the path has to be absolute;
        if the filename is not given, then the user is asked through a dialog box to select the file himself.
        :param curr_directory: (optional) A boolean flag, if True the path in filename has to be relative to the
        current directory, else it has to be absolute.
        :param max_events: The maximum number of events to load from the file.

    Returns:
        :return: A tuple (timestamps, addresses).
        timestamps - A single dimensional numpy array holding the timestamps in microseconds, of length n_events.
        addresses - A single dimensional numpy array holding the corresponding address values, of length n_events.

    """
    if filename is None:
        filename = askopenfilename()
    elif curr_directory is True:
        curr_directory = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(curr_directory, filename)

    assert (filename.endswith('.aedat') or filename.endswith('.dat')), 'The given file has to be either a ' \
                                                                       '.aedat file or a .dat file'

    token = '#!AER-DAT'
    with open(filename, 'rb') as f:
        version = 2  # default version value
        for line in iter(f.readline, ''):
            if line.startswith('#'):
                if line[:len(token)] == token:
                    version = float(line[len(token):])
                bof = f.tell()
            else:
                break

        num_bytes_per_event = 6
        if version == 1:
            num_bytes_per_event = 6
        elif version == 2:
            num_bytes_per_event = 8
        f.seek(0, 2)
        eof = f.tell()
        # noinspection PyUnboundLocalVariable
        num_events = (eof - bof) // num_bytes_per_event
        if num_events > max_events:
            num_events = max_events
        
        f.seek(bof)
        if version == 2:
            data = np.fromfile(f, dtype='>u4', count=2 * num_events)
            all_address = data[::2]
            all_timestamps = data[1::2]
        elif version == 1:
            data = np.fromfile(f, dtype='>u2', count=3 * num_events)
            all_address = data[::3]
            data_time_stamps = np.delete(data, slice(0, data.shape[0], 3))
            all_timestamps = np.fromstring(data_time_stamps.tobytes(), dtype='>u4', count=num_events)
        else:
            warnings.warn("The AER-DAT version of the current file is {}, "
                          "the loading function for which has not been implemented yet.".format(version))

    # noinspection PyUnboundLocalVariable
    return all_timestamps.astype(np.uint32), all_address.astype(np.uint32)


# noinspection PyTypeChecker
def decode_ams1b(timestamps, addresses, return_type=True, reset_time_stamps=False):
    """Decodes the timestamps and addresses extracted from an aedat file for CochleaAMS1b.

    The timestamps are converted to float32 format in seconds.
    The addresses are decoded into required attributes.
    If the return_type parameter is False, then the function returns the timestamp, channel, ear, neuron, filterbank
    id arrays in that order.
    Else if the return_type parameter is True, then the function returns timestamp, ear and the type id arrays. The type
    id array is a combined attribute based on channel, ear and filterbank. The True option for the parameter is used for
    the localization problem, where we compare the neurons across the two ears from a single type.

    Args:
        :param timestamps: The timestamps in uint32 format, decoded directly from the aerdat file.
        :param addresses: The addresses in uint32 format, decoded directly from the aerdat file.
        :param return_type: A boolean parameter to decide the format to return. If False, then the natural attributes of
        the events like timestamp, channel, ear, neuron and filterbank are returned. Else, the timestamp, ear and type
        is returned. Defaults to True.
        :param reset_time_stamps: A boolean parameter to decide if the timestamps are reset to start from 0. Defaults to
        True.

    Returns:
        :return: A tuple (timestamps, channel_id, ear_id, neuron_id, filterbank_id) if return_type is False,
        else a tuple (timestamps, ear_id, type_id).
    """
    time_wrap_mask = int("80000000", 16)
    adc_event_mask = int("2000", 16)

    address_mask = int("00FC", 16)
    neuron_mask = int("0300", 16)
    ear_mask = int("0002", 16)
    filterbank_mask = int("0001", 16)

    # temporarily remove all StartOfConversion events, no clue what this means!
    soc_idx_1 = int("302C", 16)
    soc_idx_2 = int("302D", 16)
    events_without_start_of_conversion_idx = np.where(np.logical_and(addresses != soc_idx_1, addresses != soc_idx_2))
    timestamps = timestamps[events_without_start_of_conversion_idx]
    addresses = addresses[events_without_start_of_conversion_idx]

    # finding cochlea events
    cochlea_events_idx = np.where(np.logical_and(addresses & time_wrap_mask == 0, addresses & adc_event_mask == 0))
    timestamps_cochlea = timestamps[cochlea_events_idx]
    addresses_cochlea = addresses[cochlea_events_idx]

    if reset_time_stamps:
        timestamps_cochlea = timestamps_cochlea - timestamps_cochlea[0]
    timestamps_cochlea = timestamps_cochlea.astype(np.float32) / 1e6

    # decoding addresses to get ear id, on off id and the channel id
    channel_id = np.array((addresses_cochlea & address_mask) >> 2, dtype=np.int8)
    ear_id = np.array((addresses_cochlea & ear_mask) >> 1, dtype=np.int8)
    neuron_id = np.array((addresses_cochlea & neuron_mask) >> 8, dtype=np.int8)
    filterbank_id = np.array((addresses_cochlea & filterbank_mask), dtype=np.int8)

    if return_type:
        type_id = get_type_id('ams1b', channel=channel_id, neuron=neuron_id, filterbank=filterbank_id)
        return timestamps_cochlea, ear_id, type_id

    return timestamps_cochlea, channel_id, ear_id, neuron_id, filterbank_id


# noinspection PyTypeChecker
def decode_lp(timestamps, addresses, return_type=True, reset_time_stamps=False):
    """Decodes the timestamps and addresses extracted from an aedat file for CochleaLP.

    The timestamps are converted to float32 format in seconds.
    The addresses are decoded into required attributes.
    If the return_type parameter is False, then the function returns the timestamp, channel, ear, on_off
    id arrays in that order.
    Else if the return_type parameter is True, then the function returns timestamp, ear and the type id arrays. The type
    id array is a combined attribute based on channel and on_off. The True option for the parameter is used for
    the localization problem, where we compare the neurons across the two ears from a single type.

    Args:
        :param timestamps: The timestamps in uint32 format, decoded directly from the aerdat file.
        :param addresses: The addresses in uint32 format, decoded directly from the aerdat file.
        :param return_type: A boolean parameter to decide the format to return. If False, then the natural attributes of
        the events like timestamp, channel, ear and on_off are returned. Else, the timestamp, ear and type
        is returned. Defaults to True.
        :param reset_time_stamps: A boolean parameter to decide if the timestamps are reset to start from 0. Defaults to
        True.

    Returns:
        :return: A tuple (timestamps, channel_id, ear_id, on_off_id) if return_type is False,
        else a tuple (timestamps, ear_id, type_id).
    """
    time_wrap_mask = int("80000000", 16)
    adc_event_mask = int("2000", 16)

    address_mask = int("00FC", 16)
    on_off_mask = int("0001", 16)
    ear_mask = int("0002", 16)

    # temporarily remove all StartOfConversion events, no clue what this means!
    soc_idx_1 = int("302C", 16)
    soc_idx_2 = int("302D", 16)
    events_without_start_of_conversion_idx = np.where(np.logical_and(addresses != soc_idx_1, addresses != soc_idx_2))
    timestamps = timestamps[events_without_start_of_conversion_idx]
    addresses = addresses[events_without_start_of_conversion_idx]

    # finding cochlea events
    cochlea_events_idx = np.where(np.logical_and(addresses & time_wrap_mask == 0, addresses & adc_event_mask == 0))
    timestamps_cochlea = timestamps[cochlea_events_idx]
    addresses_cochlea = addresses[cochlea_events_idx]

    if reset_time_stamps:
        timestamps_cochlea = timestamps_cochlea - timestamps_cochlea[0]
    timestamps_cochlea = timestamps_cochlea.astype(np.float32) / 1e6

    # decoding addresses to get ear id, on off id and the channel id
    channel_id = np.array((addresses_cochlea & address_mask) >> 2, dtype=np.int8)
    ear_id = np.array((addresses_cochlea & ear_mask) >> 1, dtype=np.int8)
    on_off_id = np.array((addresses_cochlea & on_off_mask), dtype=np.int8)

    if return_type:
        type_id = get_type_id('lp', channel=channel_id, on_off=on_off_id)
        return timestamps_cochlea, ear_id, type_id

    return timestamps_cochlea, channel_id, ear_id, on_off_id


# noinspection PyTypeChecker
def decode_ams1c(timestamps, addresses, return_type=True, reset_time_stamps=False):
    """Decodes the timestamps and addresses extracted from an aedat file for CochleaAMS1c.

    The timestamps are converted to float32 format in seconds.
    The addresses are decoded into required attributes.
    If the return_type parameter is False, then the function returns the timestamp, channel, ear, neuron, filterbank
    id arrays in that order.
    Else if the return_type parameter is True, then the function returns timestamp, ear and the type id arrays. The type
    id array is a combined attribute based on channel, ear and filterbank. The True option for the parameter is used for
    the localization problem, where we compare the neurons across the two ears from a single type.
    This function is essentially the same as the corresponding one for CochleaAMS1b.

    Args:
        :param timestamps: The timestamps in uint32 format, decoded directly from the aerdat file.
        :param addresses: The addresses in uint32 format, decoded directly from the aerdat file.
        :param return_type: A boolean parameter to decide the format to return. If False, then the natural attributes of
        the events like timestamp, channel, ear, neuron and filterbank are returned. Else, the timestamp, ear and type
        is returned. Defaults to True.
        :param reset_time_stamps: A boolean parameter to decide if the timestamps are reset to start from 0. Defaults to
        True.

    Returns:
        :return: A tuple (timestamps, channel_id, ear_id, neuron_id, filterbank_id) if return_type is False,
        else a tuple (timestamps, ear_id, type_id).
    """
    time_wrap_mask = int("80000000", 16)
    adc_event_mask = int("2000", 16)

    address_mask = int("00FC", 16)
    neuron_mask = int("0300", 16)
    ear_mask = int("0002", 16)
    filterbank_mask = int("0001", 16)

    # temporarily remove all StartOfConversion events, no clue what this means!
    soc_idx_1 = int("302C", 16)
    soc_idx_2 = int("302D", 16)
    events_without_start_of_conversion_idx = np.where(np.logical_and(addresses != soc_idx_1, addresses != soc_idx_2))
    timestamps = timestamps[events_without_start_of_conversion_idx]
    addresses = addresses[events_without_start_of_conversion_idx]

    # finding cochlea events
    cochlea_events_idx = np.where(np.logical_and(addresses & time_wrap_mask == 0, addresses & adc_event_mask == 0))
    timestamps_cochlea = timestamps[cochlea_events_idx]
    addresses_cochlea = addresses[cochlea_events_idx]

    if reset_time_stamps:
        timestamps_cochlea = timestamps_cochlea - timestamps_cochlea[0]
    timestamps_cochlea = timestamps_cochlea.astype(np.float32) / 1e6

    # decoding addresses to get ear id, on off id and the channel id
    channel_id = np.array((addresses_cochlea & address_mask) >> 2, dtype=np.int8)
    ear_id = np.array((addresses_cochlea & ear_mask) >> 1, dtype=np.int8)
    neuron_id = np.array((addresses_cochlea & neuron_mask) >> 8, dtype=np.int8)
    filterbank_id = np.array((addresses_cochlea & filterbank_mask), dtype=np.int8)

    if return_type:
        type_id = get_type_id('ams1b', channel=channel_id, neuron=neuron_id, filterbank=filterbank_id)
        return timestamps_cochlea, ear_id, type_id

    return timestamps_cochlea, channel_id, ear_id, neuron_id, filterbank_id


def get_type_id(sensor_type, channel=None, neuron=None, filterbank=None, on_off=None):
    """Utility function used in calculating the type_id of an event.

    The type_id of the event combines all the attributes of an event other than the ear attribute into one attribute.
    This is used so we can compare the events across the two ears but with every other attribute being the same. This
    is used when using the events for localization.

    Args:
        :param sensor_type: The sensor type in string format. Current options are 'ams1c', 'ams1b' or 'lp'.
        :param channel: Channel ids as numpy array.
        :param neuron: Neuron ids as numpy array, only if the sensor_type is 'ams1b' or 'ams1c'.
        :param filterbank: Filterbank ids as numpy array, only if the sensor_type is 'ams1b' or 'ams1c'.
        :param on_off: On_off ids as numpy array, only if the sensor_type is 'lp'.

    Returns:
        :return: A numpy array.
    """
    if sensor_type == 'ams1b' or sensor_type == 'ams1c':
        type_id = channel + NB_CHANNELS * neuron + NB_CHANNELS * NB_NEURON_TYPES * filterbank
        return type_id
    elif sensor_type == 'lp':
        type_id = channel + NB_CHANNELS * on_off
        return type_id
    else:
        warnings.warn('The sensor type is not implemented yet.')


def separate_streams(timestamps, itds, num_streams=12):
    """Utility function used to separate streams based of timestamps.

    Based on the given number of streams in the recording, through the parameter num_streams, the itds are separated
    into streams, based on the timestamps into equal time lengths. Used to separate the streams in the recordings at
    the University Hospital in August 2017 for the localization project. Returns list of itds in the separated streams.

    Realised it is not a good way to separate streams. That's because the timestamps do not start at zero and end at
    the time you think the recording ended, so could not rely on the timestamps to divide time.

    Args:
        :param timestamps: The timestamps, in np.float32 format.
        :param itds: The itds, in np.float32 format.
        :param num_streams: An integer, representing the number of streams to be separated. Defaults to 12.

    Returns:
        :return: A list of np.float32 arrays, each corresponding to the itds in the separated streams.
    """
    itd_streams = []
    min_timestamp, max_timestamp = np.amin(timestamps), np.amax(timestamps)
    time_length = (max_timestamp - min_timestamp) / num_streams
    for idx in range(num_streams):
        indices = np.where((timestamps > min_timestamp + idx * time_length) &
                           (timestamps < min_timestamp + (idx + 1) * time_length))[0]
        itds_to_append = itds[indices]
        itd_streams.append(itds_to_append)
    return itd_streams


def get_labels(timestamps, num_streams=12):
    """Get the label for every event based on time based stream separation.

    Based on the given number of streams in the recording, through the parameter num_streams, the itds are separated
    into streams, based on the timestamps into equal time lengths. Used to separate the streams in the recordings at
    the University Hospital in August 2017 for the localization project. Returns a label representing the separated
    stream for every event.

    Realised it is not a good way to separate streams. That's because the timestamps do not start at zero and end at
    the time you think the recording ended, so could not rely on the timestamps to divide time.

    Args:
        :param timestamps: The timestamps, in np.float32 format.
        :param num_streams: An integer, representing the number of streams to be separated. Defaults to 12.

    Returns:
        :return: A numpy array with the same shape as timestamps, carrying the labels representing the separated stream.
    """
    labels = np.zeros_like(timestamps)
    min_timestamp, max_timestamp = np.amin(timestamps), np.amax(timestamps)
    time_length = (max_timestamp - min_timestamp) / num_streams
    for idx in range(num_streams):
        indices = np.where((timestamps > min_timestamp + idx * time_length) &
                           (timestamps < min_timestamp + (idx + 1) * time_length))[0]
        labels[indices] = idx
    return labels


def separate_streams_v2(itds, timestamps, num_streams, angles, prior_angles, speech_time, silence_time):
    """Utility function used to separate streams based of timestamps.

    Based on the number of streams, speech time in each stream and silence time between streams, the itd streams are
    separated.

    Args:
        :param itds: The itd numpy array.
        :param timestamps: The timestamp numpy array.
        :param num_streams: Number of streams in the event stream.
        :param angles: The location (in angle with respect to the zero itd in degrees) at each of the stream.
        :param prior_angles: The angles for which the priors are to be calculated.
        :param speech_time: The speech time in each stream.
        :param silence_time: The silence time between streams.

    Returns:
        :return: List of itd streams.
    """
    itd_streams = [[] for angle in prior_angles]
    for stream in range(num_streams):
        index = get_prior_index(angles[stream])
        start_time = stream * (speech_time + silence_time)
        end_time = (stream + 1) * speech_time + stream * silence_time
        indices = np.where((timestamps >= start_time) & (timestamps < end_time))[0]
        itd_streams[index].extend(itds[indices])
    return itd_streams


def get_prior_index(angle):
    """Gets the index of an angle with respect to the prior angles.

    Args:
        :param angle: The angle in degrees.

    Returns:
        :return: An integer index.
    """
    if 0 <= angle <= 90:
        equivalent_angle = angle
    elif angle <= 270:
        equivalent_angle = 180 - angle
    else:
        equivalent_angle = angle - 360
    index = int((equivalent_angle + 90) / 30)
    return index

if __name__ == '__main__':
    test_timestamps, test_addresses = loadaerdat()
    test_timestamps, test_ears, test_types = decode_ams1b(test_timestamps, test_addresses)
    print('Done')
