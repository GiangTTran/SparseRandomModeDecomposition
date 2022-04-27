# SRMD: Sparse Random Mode Decomposition

## Introduction

A Python implimentation of Sparse Random Mode Decomposition

The goal of this algorithm is to provide a sparse time-frequency representation
of a signal to decompose the signal into intrinsic mode-like functions.

## Installation

Easiest method:

> \$ pip install git+https://github.com/GiangTTran/SparseRandomModeDecomposition.git#egg=srmdpy

The code can also be accessed by cloning this repository

> \$ git clone <https://github.com/GiangTTran/SparseRandomModeDecomposition>

or downloading the source by clicking `Code -> Download ZIP`, and then installed via

> \$ python setup.py install

## The Algorithm

The algorithm can be broken into three main steps:
    
1. Randomly generate many time-frequency features
2. Find a sparse representation of the input with respect to the random features
3. Cluster near-by features into modes

The features used are (Gaussian) windowed sinusoids with random phases, although
any wavelet-like feature could be used. The sparse representation is found by
solving the associated Basic Pursuit Denoising Problem (BPDP) using the SPGL1
algorithm. And the clustering is performed by scikit-learn's implimentation of
the DBSCAN algorithm.

Given a time series ```y=[y1, ..., ym]``` sampled at time points ```t=[t1, ..., tm],```
SRMD recovers the modes ```s1, ..., sK``` and possibly denoise the input signal where

```y(t) = s1(t) + s2(t) + ... + sK(t) + noise.```

This method is most effective when the modes sk(t) occupy connected curves
or regions that are mutualy disjoint in time-frequency space such as
non-intersecting intrinsic mode functions.

Specificly, SRMD(y, ...) first solves the problem of representing a time-series
y as a sum of random features

```y(t) ~ sum_j(c_j * exp(-0.5 * ((t-tau_j)/w)^2) * sin(2*pi*frq_j*t + phs_j)),```

where the learned weights vector c is sparse, by solving BPDN

```argmin ||c||_1 s.t. ||Ac - y||_2 < r*||y||_2.```

Here, ```(A)_ij = exp(-0.5 * ((t_i-tau_j)/w)^2) * sin(2*pi*frq_j*t_i + phs_j).```

The features are clustered with DBSCAN which groups features with near-by
```(tau_j, frq_j*frq_scale)``` in terms of their l2 distance to each other.

## ```srmdpy``` File Overview

srmdpy.py:
```
SRMD(y, t):
    Decomposes a time series y(t) = sum_k s_k(t) into modes s_k(t) that are well
    connected in time-frequency space, and mutualy disjoint from each other.
```
    
utils.py: 
Hosts the helper functions for generating the features used by SRMD
```
generate_features(N, t):
    Generates N random features on the time points t.
    
window(t, w):
    Creates a Gaussian window function of width w on the time points t.
```

visualization.py:
Contains useful plotting functions for visualizating decomposition results.
```
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
```

constants.py:
Store helpful constants and default values used by SRMD and helper functions.
```
twopi = 2 * pi
default_w = 0.1
default_r = 0.05
default_min_samples = 4
```
## ```examples``` File Overview

minimal.ipynb:
Decomposes a composite signal with two pure sinusoids at 5 Hz and 20 Hz into its two modes.

frequency_estimation.ipynb:

graviational.ipynb:

music.ipynb:
use SRMD_music()

discontinuous.ipynb:

intersecting.ipynb:
Show all the hyperparameters

## ```data``` Files
synthetic_data.py:
Creates the synthetic data from the SRMD paper

music:
flute.wav, guitar.wav, both.wav

gravitational?:

## About

Created by Nicholas Richardson March 2022 based on the decomposition algorithm
developed by Nicholas Richardson, Hayden Schaeffer, and Giang Tran.

### Contact

Feel free to reach out to njericha@uwaterloo.ca if you have any questions or
just to say *hi*. Any suggestions and contributions are also welcome!

### Citation
If you found this package useful in any way, please cite the repository as:

```latex
@misc{srmdpy,
  author = {Richardson, Nicholas},
  title = {Python implimentation of the Sparse Random Mode Decomposition algorithm},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/GiangTTran/SparseRandomModeDecomposition}},
  version = {0.0.1}
}
```

and see the [relevant paper](https://arxiv.org/abs/2204.06108):

```latex
@article{richardson2022srmd,
  doi = {10.48550/ARXIV.2204.06108},
  url = {\url{https://arxiv.org/abs/2204.06108}},
  author = {Richardson, Nicholas and Schaeffer, Hayden and Tran, Giang},
  title = {{SRMD}: Sparse Random Mode Decomposition},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license},
  journal = {arXiv preprint arXiv:2204.06108}
}
```
