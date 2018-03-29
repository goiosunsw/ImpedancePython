import numpy as np
import matplotlib.pyplot as pl
import sys
import os
import scipy.io as sio

from pympedance.UNSW import *
from pympedance.plot_utils import bodeplot

pl.style.use('ggplot')

# Define path for the data
script_path, _ = os.path.split(os.path.realpath(__file__))
print(script_path)

# The actual impedance measurement file
meas_file = os.path.join(script_path, '../tests/data/i.mat')

# homedir = os.environ['HOME']
# basedir = os.path.join(homedir,'/home/goios/Data/20180221-26mmHead/')
# datadir = os.path.join(basedir,'meas')
# meas_file = os.path.join(datadir,'/home/goios/Data/20180221-26mmHead/meas/OpenPipe170mm.mat')

# Read the impedance file with its parameters
io = ImpedanceMeasurement(meas_file)

# Plot it
fig,ax = bodeplot(io.f,io.z,lw=3)

# Recalculate the calibration matrix
mic_pairs, pair_sets = io.get_mic_pair_set()

for pair, pset in zip(mic_pairs, pair_sets):
    # Plot the new impedance on top of the first one
    bodeplot(pset.f,pset.z, ax=ax, label=str(pair),lw=1)
pl.legend()
pl.show()




