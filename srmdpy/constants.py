"""Constants and defaults for SRMD

Store helpful constants and default values used by SRMD and helper functions
"""
__all__ = ['twopi', 'default_w', 'default_r', 'default_min_samples']

import numpy as np

# Helpful constants
twopi = 2 * np.pi

# Defaults
default_w = 0.1
default_r = 0.05
default_min_samples = 4