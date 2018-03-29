import numpy as np
import matplotlib.pyplot as pl
import sys
import os
import scipy.io as sio

from pympedance.UNSW import *
from pympedance.plot_utils import bodeplot

# Define path for the data
script_path, _ = os.path.split(os.path.realpath(__file__))
print(script_path)

# The actual impedance measurement file
meas_file = os.path.join(script_path, '../tests/data/i.mat')

# The calibration files corresponding to the above measurement
calib_infimp_file = os.path.join(script_path, '../tests/data/InfImpCalib.mat')
calib_infpipe_file = os.path.join(script_path, '../tests/data/InfPipeCalib.mat')

# Read the impedance file with its parameters
io = ImpedanceMeasurement(meas_file)

# Plot it
fig,ax = bodeplot(io.f,io.z)

# Recalculate the calibration matrix
old_a = io.parameters.A
new_a = io.parameters.calc_calibration_marix(infinite_imp_file=calib_infimp_file,
                                             infinite_pipe_file=calib_infpipe_file)

# Set the matrix in parameters
# (will probably be done automatically in a later version)
io.parameters.A = new_a
# recalculate the impedance with the new calibration
new_z = io.calculate_impedance()

# Plot the new impedance on top of the first one
bodeplot(io.f,new_z, ax=ax)
pl.show()




