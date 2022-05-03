"""Visualization of SRMD results

Contains useful plotting functions for visualizating decomposition results.

signal(t, y):
    Plots a time series y sampled at time points t.

all_modes(t, modes):
    Plots each mode_i = modes[:, i] sampeled at t.

modes_with_cluster(t, modes, tau, frq, labels):
    Plots each mode_i and the time-frequency pairs (tau_n, frq_n) coloured
    according to labels.

weights(tau, frq, weights):
    Creates a spectrogram-like plot where time-frequency pairs (tau_n, frq_n)
    are coloured accoring to weights_n.
"""

__all__ = ['signal',
           'all_modes',
           'modes_with_cluster',
           'weights']

import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

def make_colours(n):
    """Make n unique colours.

    A mini helper to select the best colour palette given the number of unique
    colours needed n.
    """
    if n <= 4:
        cp = sns.color_palette("colorblind")
        colours = cp[1:3] + cp[4:5] + cp[0:1] # give preference to these colours
    elif n <= 10:
        colours = sns.color_palette("colorblind")
    else:
        # modes modes than colours in the "colorblind" palette,
        # need to use evenly spaced colours on the colour wheel
        colours = sns.color_palette("husl", n)

    return colours

def signal(t, y, title=None, **kwargs):
    """Basic plotting of a function

    Plots a time series y sampled at points t on a new figure.
    """
    fig = plt.figure()
    plt.plot(t, y, **kwargs)
    if title:
        plt.title(title)
    plt.xlabel('time')
    plt.ylabel('signal')
    plt.show()

def all_modes(t, modes, title='',**kwargs):
    """Plots each mode in a seperate figure

    modes.shape[0] should equal len(t), where modes are stored row-wise
    """
    for i in range(modes.shape[1]):
        signal(t, modes[:,i], title= title+f' {i+1}',**kwargs)

def modes_with_cluster(t, modes, tau, frq, labels, **kwargs):
    """Plots the time-frequency clusters with their matching modes.

    The learned time shifts and frequencies are plotted so each cluster has a
    unique colour. The recovered modes from each cluster are plotted with the
    same colour as the cluster.
    
    See top right plot of Figure 1 of SRMD paper for an example of the time-
    frequency clusters, and the middle row in Figure 1 for an example of
    plotting the modes with matching colours.
    """
    n_modes = modes.shape[1]
    colours = make_colours(n_modes)

    fig = plt.figure()
    list_of_labels = list(set(labels)) # ensure same ordering when looping
    for l in list_of_labels:
        if l == -1:
            continue
        mode_index = np.equal(labels, l)
        taus = tau[mode_index]
        frqs = frq[mode_index]
        plt.scatter(taus, frqs, color=colours[l])

    plt.title('Time-Frequency Clusters Identified')
    plt.xlabel('time')
    plt.ylabel('frequency')
    plt.show()

    for l in list_of_labels:
        if l == -1:
            continue
        signal(t, modes[:,l], title=f'Learned Mode {l+1}', c=colours[l])

def weights(tau, frq, weights, labels=None, title=None, **kwargs):
    """Plots the magnitude of nonzero weights in time frequency space

    If labels are given, features that were labeled -1 (outliers) will be
    ignored. Lines and curves should reveal the instentaneous frequency of the
    input signal.
    
    See top left plot of Figure 1 in SRMD paper for an example.
    """
    kwargs['cmap'] = sns.color_palette('crest', as_cmap=True) #'viridis'
    kwargs['norm'] = matplotlib.colors.LogNorm()
    weights = np.abs(weights)

    fig = plt.figure()

    if labels is not None:
        ax = plt.scatter(tau[labels != -1], frq[labels != -1],
                    c=weights[labels != -1], **kwargs)
    else:
        ax = plt.scatter(tau, frq, c=weights, **kwargs)

    if title:
        plt.title(title)

    plt.xlabel('time')
    plt.ylabel('frequency')
    cbar = plt.colorbar(ax)
    cbar.set_label("weight's magnitude")
    plt.show()
