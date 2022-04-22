"""Helpers for SRMD

Hosts the helper functions for generating the features used by SRMD

generate_features(N, t):
    Generates N random features on the time points t.

window(t, w):
    Creates a Gaussian window function of width w on the time points t.
"""

__all__ = ['generate_features', 'window']

import numpy as np

from .constants import twopi, default_w, default_r, default_min_samples

def generate_features(N, t, max_frq=None, w=default_w, seed=None):
    """Generates N random features.

    Given the desired number of features N, generates windowed sinusoidal
    features with random time-shifts, frequencies, and phases evaluated at the
    time points t.

    Inputs
    ------
    N : int
        The number of features to generate.
    t : numpy array
        The time points to compute the value of the features

    Parameters
    ----------
    max_frq : float, (default: None)
        Maximum possible frequency a feature could have. If None is given, will
        use half the sample rate: 0.5 * (len(t)+1) / (t[-1] - t[0])
    w : float, (default: 0.1)
        Window size of the features in seconds. Defaults to 0.1s or 100ms.
    seed : int, (default: None)
        Seed to use in the random generation of the features. This is useful for
        repeatability. If None is given, a random seed will be used.

    Outputs
    -------
    features : numpy array, features.shape == (m, N)
        The value of the features at time points t.
    (tau, frq, phs) : tuple of numpy arrays,
        Time-shifts, frequencies, and phases of the features used in the
        representation of the input y. Arrays have shape (N,).
    """
    # Constants and argument parsing
    L = t[-1] - t[0]

    if max_frq is None:
        m = len(t)
        max_frq = 0.5 * (m+1) / L  # using m+1 to be slightly above Nyquist rate

    # Generate random times, frequencies, and phases
    rng = np.random.default_rng(seed)
    tau = rng.random((1, N)) * L + t[0]
    frq = rng.random((1, N)) * max_frq
    phs = rng.random((1, N)) * twopi

    _t = np.reshape(t, (-1, 1))

    features = window(_t - tau, w) * np.sin(twopi*frq*_t + phs)

    # Reshape from (1, N) to (N,)
    tau = tau.squeeze()
    frq = frq.squeeze()
    phs = phs.squeeze()

    return  features, (tau, frq, phs)

def window(t, w=default_w):
    """Creates a truncated gaussian window.

    The window has a maximum value of 1, mean at 0, and standard deviation w/2.
    Time points outside of three standard deviations are set to zero.

    Parameters
    ----------
    t : numpy array
        Time points to evaluate the window.
    w : float
        Width of the gaussian window or twice its standard deviation. Should be
        given in the sames units as t and not number of samples.

    Returns
    -------
    numpy array
        The values of the window at t
    """
    # Standard deviation s. Half of the window width w
    s = w / 2

    # Zero time values outside three standard deviations
    domain = (np.sign(3*s + t) + np.sign(3*s - t)) / 2

    return np.exp(-0.5 * (t/s)**2) * domain