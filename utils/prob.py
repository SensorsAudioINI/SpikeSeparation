from __future__ import division

from scipy.stats import norm
import warnings
import numpy as np
import progressbar
from matplotlib import pyplot

index_angles_01 = np.array([[12, 0], [11, -30], [9, -90], [8, -60], [4, 60], [3, 90], [1, 30]])
index_angles_02 = np.array([[1, 30], [2, 60], [3, 90], [4, 120], [5, 150], [6, 180], [7, 210], [8, 240], [9, 270],
                            [10, 300], [11, 330], [12, 0]])


def get_estimates(itds, initial_estimate, transition_probabilities, itd_dict, prior, save_to_file=None, verbose=False):
    """Implements the basic probabilistic model.

    Args:
        :param itds: The itds as a numpy array, of dtype np.float32, in seconds.
        :param initial_estimate: The initial estimate as numpy array of size num_possible_locations. Note that the array
        should be a valid probability distribution, so should sum upto 1.
        :param transition_probabilities: The transition probabilities, as a numpy 2D array. Again, the rows must be
        valid probability distributions.
        :param itd_dict: The itd mapping between the quantized itds and their indices in array format.
        :param prior: The prior distributions, as numpy 2D array, rows should be valid probability distributions.
        :param save_to_file: If not None, filename is expected, to which the estimates and argmax_estimates are saved.
        :param verbose: If True, then a progressbar display of the progress will be displayed.

    Returns:
        :return: A tuple (estimates, argmax_estimates)
        estimates: A numpy 2D array, with the probability estimates at every itd.
        argmax_estimates: A numpy array, with the argmax of the probability estimate at every itd.
    """
    localization_estimate = initial_estimate
    num_itds = len(itds)
    estimates = np.zeros(shape=(num_itds, prior.shape[0]), dtype=np.float32)
    argmax_estimates = np.zeros(shape=num_itds, dtype=np.int32)
    bar = progressbar.ProgressBar() if verbose else identity
    for itd_idx, itd in bar(enumerate(itds)):
        position_matrix = np.multiply(transition_probabilities, localization_estimate)
        position_probability = np.sum(position_matrix, axis=1)
        motion_probability = np.array([prior[idx][np.argmin(np.abs(itd_dict - itd))] for idx in range(prior.shape[0])])
        probability_to_normalize = np.multiply(motion_probability, position_probability)
        localization_estimate = probability_to_normalize / sum(probability_to_normalize)
        estimates[itd_idx] = localization_estimate
        argmax_estimates[itd_idx] = np.argmax(localization_estimate)
        if np.isnan(np.sum(localization_estimate)):
            warnings.warn('Something wrong with the estimate.')
    if save_to_file is not None:
        np.savez(save_to_file, estimates=estimates, argmax_estimates=argmax_estimates)
    return np.array(estimates, dtype=np.float32), np.array(argmax_estimates, dtype=np.float)


def get_priors(itd_streams, max_itd=800e-6, num_bins=80, save_to_file=None):
    """Calculate prior distributions based on separated itd_streams.

    Args:
        :param itd_streams: A list of separated itd streams, one for each discrete location.
        :param max_itd: The max_itd parameter for the itd algorithm, in seconds.
        :param num_bins: The number of bins the itds are quantized into.
        :param save_to_file: If not None, filename is expected, to which the prior distribution is saved.

    Returns:
        :return: The priors, a 2D numpy array, with each row corresponding to a location.
    """
    priors = np.zeros(shape=(len(itd_streams), num_bins), dtype=np.float32)
    for idx, itd_stream in enumerate(itd_streams):
        hist = np.histogram(itd_stream, bins=num_bins, range=(-max_itd, max_itd))[0] / len(itd_stream)
        priors[idx] = hist
    if save_to_file is not None:
        np.save(save_to_file, priors)
    return priors


def get_transition_probabilities(index_angles=index_angles_01, sigma=5):
    """Get Gaussian transition probabilities based on angles and a given sigma.

    Args:
        :param index_angles: The list of tuples with location index and the corresponding angle.
        :param sigma: The sigma for the Gaussian distributions.

    Returns:
        :return:  A numpy 2D array.
    """
    transition_probabilities = np.zeros(shape=(len(index_angles), len(index_angles)), dtype=np.float32)
    angles_original = index_angles[:, 1]
    angles = np.sort(angles_original)
    for angle_index, index_angle in enumerate(index_angles):
        mean = index_angle[1]
        angle_distribution = norm(mean, sigma).pdf(angles)
        angle_dict = {}
        for idx, angle in enumerate(angles):
            angle_dict[angle] = angle_distribution[idx]
        angle_distribution = [angle_dict[angle] for angle in angles_original]
        transition_probabilities[angle_index] = angle_distribution
    return transition_probabilities


def moving_average(estimates, window_length=10):
    """Implements moving window average to smoothen the estimates.

    Args:
        :param estimates: The estimates from the probabilistic model.
        :param window_length: The window length for the smoothing.

    Returns:
        :return: The smoothed estimates, as a numpy array.
    """
    averaged_estimates = np.zeros_like(estimates)
    for idx in range(len(estimates) - window_length + 1):
        averaged_estimates[idx] = np.mean(estimates[idx:idx + window_length], axis=0)
    for idx in range(len(estimates) - window_length + 1, len(estimates)):
        averaged_estimates[idx] = averaged_estimates[len(estimates) - window_length]
    return averaged_estimates


def identity(x):
    return x


def get_kalman_estimates(itds, h_k=-178., r_k=210. ** 2, f_k=1., q_k=(0.05) ** 2,
                         init_state=np.array(0), init_var_state=np.array(0) ** 2,
                         version='basic', itd_shift=37.20):
    itds = itds * 1e6 + itd_shift
    estimates, variances = [], []
    x_k_k = init_state
    p_k_k = init_var_state
    for itd in itds:
        x_k_km = f_k * x_k_k
        p_k_km = f_k * p_k_k * f_k + q_k
        y_k = itd - h_k * x_k_km
        s_k = r_k + h_k * p_k_km * h_k
        k_k = p_k_km * h_k / s_k
        x_k_k = x_k_km + k_k * y_k
        p_k_k = p_k_km - k_k * h_k * p_k_km
        y_k_k = itd - h_k * x_k_k
        estimates.append(x_k_k)
        variances.append(p_k_k)
    return np.array(estimates), np.array(variances)


if __name__ == '__main__':
    test_index_angles = np.array([[12, 0], [11, -30], [9, -90], [8, -60], [4, 60], [3, 90], [1, 30]])
    test_transition_probabilities = get_transition_probabilities(test_index_angles, sigma=5)
    pyplot.imshow(test_transition_probabilities, aspect='auto', interpolation='nearest')
    pyplot.show()
    print('Hello world, nothing to test for now.')
