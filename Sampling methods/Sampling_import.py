import Sampling_regimes
import numpy as np
import os
import sys

###this is the function to grab the correct sampling mask asked for in the reconstruction file
def get_sampling_mask(sampling_regime, kspace, percent_sampling, output_file):
    ##kspace is taken in after resizing has happened, needs to be called to get the correct size
    dimviews, dimsliceviews, dimslices = kspace.shape[0:3]

    ##list of regimes:
    #
