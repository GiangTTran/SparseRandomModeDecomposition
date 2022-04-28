"""Sparse Random Mode Decomposition in Python

Hosts the Sparse Random Mode Decomposition algorithm

SRMD(y, t):
    Decomposes a time series y(t) = sum_k s_k(t) into modes s_k(t) that are well
    connected in time-frequency space, and mutualy disjoint from each other.
"""

__all__ = ['SRMD']

import logging

import numpy as np
from spgl1 import spg_bpdn
from sklearn.cluster import DBSCAN

from .utils import generate_features
from .constants import twopi, default_w, default_r, default_min_samples

def SRMD(y, t=None, N_features=None, eps=None, *, max_frq=None, w=default_w,
         r=default_r, threshold=None, frq_scale=None, min_samples=default_min_samples,
         seed=None, n_modes=None, verbosity=0, return_features=False, cutoff=None):
    """Implimentation of the Sparse Random Mode Decomposition algorithm

    **Sparse Random Mode Decomposition**

    Given a time series y=[y1, ..., ym] sampled at time points t=[t1, ..., tm],
    recover the modes s1, ..., sK and possibly denoise the input signal where

    y(t) = s1(t) + s2(t) + ... + sK(t) + noise.

    This method is most effective when the modes sk(t) occupy connected curves
    or regions that are mutualy disjoint in time-frequency space such as
    non-intersecting intrinsic mode functions.

    Method
    ------
    SRMD(y, ...) solves the problem of representing y as a sum of random
    features

    y(t) ~ sum_j(c_j * exp(-0.5 * ((t-tau_j)/w)^2) * sin(2*pi*frq_j*t + phs_j)),

    where the learned weights vector c is sparse, by solving BPDN

    argmin ||c||_1 s.t. ||Ac - y||_2 < r*||y||_2.

    Here, (A)_ij = exp(-0.5 * ((t_i-tau_j)/w)^2) * sin(2*pi*frq_j*t_i + phs_j).

    The features are clustered with DBSCAN which groups features with near-by
    (tau_j, frq_j*frq_scale) in terms of their l2 distance to each other.

    Inputs
    ------
    y : numpy array
        The input signal.
    t : numpy array, (default: None)
        Time points the signal y was sampled on. If None is given, will assume
        equally spaced points on the unit interval, end points inclusive.

    Returns
    -------
    modes : numpy array, modes.shape == (m, n_modes_recovered)
        List of modes y1, ..., yn recovered from the decomposition algorithm as
        time series.
    (tau, frq, phs) : tuple of numpy arrays
        Time-shifts, frequencies, and phases of the features used in the
        representation of the input y.
    weights : numpy array
        Learned coefficients of the features used in the representation of y.
    labels : numpy array
        Which mode each feature belongs to. Modes are labeled in decreasing L2
        norm order. A label of -1 corresponds to features thrown out by DBSCAN.

    Hyperparameters
    ---------------
    N_features : int, (default: None)
        Number of features to generate. If None is given, will use 10*len(y)
    eps : float, (default: None)
        Radius of a feature's neighbourhood in time-frequency space. If None is
        given, will default to: 0.2 * (t[-1] - t[0])
    max_frq : float, (default: None)
        Maximum possible frequency a feature could have. If None is given, will
        use half the sample rate: 0.5 * (len(t)+1) / (t[-1] - t[0])
    w : float, (default: 0.1)
        Window size of the features in seconds. Defaults to 0.1s or 100ms.
    r : float, (default: 0.05)
        Maximum relative error in the representation of the signal. Defaults to
        5%. Should be between [0.01, 0.5] for sensible results.
    threshold : float, (default: None)
        Bottom percentile of nonzero-coefficients to prune after the
        coefficients are learned. If None is given, will skip this step. Should
        be in the range [0,100].
    frq_scale : float, (default: None)
        Amount to scale the frequencies of the features before the clustering
        algorithm. If None is given, will default to: (t[-1] - t[0]) / max_frq
    min_samples : int, (default: 4)
        Number of features in a neighbourhood required to be considered a core
        point in the clustering algorithm. Should be 3, 4, or 5 for sensible
        results.
    seed : int, (default: None)
        Seed to use in the random generation of the features. This is useful for
        repeatability. If None is given, a random seed will be used.
    n_modes : int, (default: None)
        Number of modes in the input signal if known. Will merge extra modes
        with the smallest L2-norm so at most n_modes are returned. If None
        is given, will not merge any modes.
    verbosity : int, (default: 0)
        If 1 will print out feature info throughout the execution of the
        function. If 2, will print out progress and parameters in addition to
        feature info. Defaults to 0 (no printing).
    return_features : bool, (default: False)
        If True, will return the learned modes in addition to weights,
        tau, frq, and phs of the features, and the features' label (which mode
        each feature belongs to).
    cutoff : float, (default: None)
        If given, will *not* use DBSCAN to cluster features and instead separate
        features into two modes: features with frequency above and below cutoff.

    Example
    -------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from srmdpy import SRMD
    >>> t = np.linspace(0,1,num=200)
    >>> y = np.cos(2*np.pi*5*t) + np.sin(2*np.pi*20*t)
    >>> kargs = {'eps':1, 'frq_scale':1, 'seed':314}
    >>> modes = SRMD(y, t, **kargs)
    >>> for i in range(modes.shape[1]):
    >>>     mode_i = modes[:,i]
    >>>     plt.figure(i)
    >>>     plt.title(f'Mode {i+1}')
    """
    # Verbosity handeling
    if verbosity == 2:
        logging_level = logging.DEBUG
    elif verbosity == 1:
        logging_level = logging.INFO
    elif verbosity == 0:
        logging_level = logging.WARNING
    else:
        raise ValueError(f'verbosity is set to {verbosity}, expected 0, 1, or 2')
    logging.basicConfig()
    log = logging.getLogger("SRMD")
    log.setLevel(logging_level)

    # Value checking for t
    if t is None:
        t = np.linspace(0,1,num=len(y))
    elif len(y) != len(t):
        raise ValueError('Input signal and time array have mismatched sizes: '
                        f'{len(y)} and {len(t)}')

    # Define useful constants
    m = len(y)       # Number of data points
    L = t[-1] - t[0] # Length of signal in time

    # Default parameter Handeling
    if N_features is None:
        N_features = 10 * m

    if max_frq is None:
        max_frq = 0.5 * (m+1) / L  # using m+1 to be slightly above Nyquist rate

    if eps is None:
        eps = 0.2 * L

    if frq_scale is None:
        frq_scale = L / max_frq

    # A nice Python >=3.8 solution...
    #for variable in {N_features, max_frq, eps, frq_scale, w, r, min_samples,
    #                 threshold, n_modes, n_modes, seed, m, L}:
    #    log.debug(f'{variable=}')

    # Record parameters
    log.debug(f'N_features = {N_features}')
    log.debug(f'max_frq = {max_frq}')
    log.debug(f'eps = {eps}')
    log.debug(f'frq_scale = {frq_scale}')
    log.debug(f'w = {w}')
    log.debug(f'r = {r}')
    log.debug(f'min_samples = {min_samples}')
    log.debug(f'threshold = {threshold}')
    log.debug(f'n_modes = {n_modes}')
    log.debug(f'seed = {seed}')
    log.debug(f'm = {m}')
    log.debug(f'L = {L}')

    # Generate random features
    log.debug('Generating random features...')
    features, (tau, frq, phs) = generate_features(N_features, t, w=w, seed=seed,
                                                  max_frq=max_frq)
    log.debug('...done!')

    # Represent y sparsely in terms of the features
    log.debug('Representing y sparsely in terms of the features...')
    sigma = r * np.linalg.norm(y)
    weights, _, _, _ = spg_bpdn(features, y, sigma)
    weights = weights.squeeze() # convert shape from (N,1) to (N,)
    log.debug('...done!')

    # Optional thresholding step
    abs_wghts = np.abs(weights)
    if threshold:
        # Only keep features above the threshold
        gate = np.percentile(abs_wghts[abs_wghts != 0], threshold)
        keep_index = abs_wghts >= gate
    else:
        # Only keep non-zero weighted features
        keep_index = np.not_equal(abs_wghts, 0)

    # Extract desired features
    tau = tau[keep_index]
    frq = frq[keep_index]
    phs = phs[keep_index]
    features = features[:,keep_index]
    weights = weights[keep_index]

    log.info(f'There are {len(weights)} nonzero features out of {N_features} '
             f'features or {len(weights)/N_features:.3%}')
    
    # Cluster features
    if cutoff:
        # Label features with frequency above cutoff 0, and below cutoff 1
        log.debug(f'Labelling features by cutoff frequency {cutoff}')
        labels = np.zeros(m).astype(int)
        labels[frq < cutoff] = 1
    else: # default clustering
        # Cluster near-by features in tau-frq space
        log.debug('Clustering near-by features in tau-frq space...')
        X = np.column_stack((tau,frq*frq_scale)) # Package tau-frq into a 2 column matrix
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit(X).labels_
        log.debug('...done!')

    # Extract modes by label
    n_labels = len(set(labels)) - (1 if -1 in labels else 0)
    modes = np.zeros((m, n_labels))
    for i in set(labels):
        if i == -1:
            # Skip features thrown out by DBSCAN
            continue
        mode_index = np.equal(labels, i)
        modes[:, i] = features[:,mode_index] @ weights[mode_index]

    # Sort modes by their norm in decreasing order
    norms = np.linalg.norm(modes, axis=0)
    sort_order = np.argsort(norms)[::-1]
    modes = modes[:,sort_order]

    # Relabel features to match new order
    re_label = {k:v for v, k in enumerate(sort_order)}
    re_label[-1] = -1
    labels = np.array([re_label[l] for l in labels])

    # Merge (sum) extra modes
    if n_modes and n_labels > n_modes:
        modes = np.hstack((modes[:,:n_modes-1],
                    np.sum(modes[:,n_modes-1:], axis=1, keepdims=True)))
        labels = np.array([(l if l < n_modes else n_modes - 1) for l in labels])

    if return_features:
        return modes, (tau, frq, phs), weights, labels
    else:
        return modes
