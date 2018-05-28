import numpy as np
import matplotlib.pyplot as pl
import sys
import os
import scipy.io as sio
import argparse as ap

parser = ap.ArgumentParser()

parser.add_argument('filename', nargs='?', help='measurement filename')
parser.add_argument('-p', '--pipe', help='set infinite pipe calibration')
parser.add_argument('-i', '--impedance', help='set infinite impedance calibration')
parser.add_argument('-f', '--flange', help='set infinite flange calibration')
args = parser.parse_args()

#import pdb
#pdb.set_trace()

from pympedance.UNSW import *
from pympedance.plot_utils import bodeplot

# Define path for the data
script_path, _ = os.path.split(os.path.realpath(__file__))
print(script_path)

# The actual impedance measurement file
if args.filename:
    meas_file = args.filename
else:
    meas_file = os.path.join(script_path, '../tests/data/i.mat')

print('Using measurement file: '+meas_file)

# Read the impedance file with its parameters
io = ImpedanceMeasurement(meas_file)

#calibs = io.parameters.find_calib_files()
calibs = io.parameters.calib_files
print('Calibration dict:')
print(calibs)

# The calibration files corresponding to the above measurement
if args.impedance:
    calib_infimp_file = args.impedance
else:
    calib_infimp_file = calibs['inf_imp']
    #calib_infimp_file = os.path.join(script_path, '../tests/data/InfImpCalib.mat')
if args.pipe:
    calib_infpipe_file = args.pipe
else:
    calib_infpipe_file = calibs['inf_pipe']
    #calib_infpipe_file = os.path.join(script_path, '../tests/data/InfPipeCalib.mat')
if args.flange:
    calib_infflange_file = args.flange
else:
    calib_infflange_file = calibs['inf_flange']
    #calib_infpipe_file = os.path.join(script_path, '../tests/data/InfPipeCalib.mat')
calib_infimp_file = None

# Plot it
fig,ax = bodeplot(io.f,io.z)

# Recalculate the calibration matrix
old_a = io.parameters.A
new_a = io.parameters.calc_calibration_matrix(infinite_imp_file=calib_infimp_file,
                                             infinite_pipe_file=calib_infpipe_file,
                                             infinite_flange_file=calib_infflange_file)

# Set the matrix in parameters
# (will probably be done automatically in a later version)
io.parameters.A = new_a
# recalculate the impedance with the new calibration
new_z = io.calculate_impedance()

# Plot the new impedance on top of the first one
bodeplot(io.f,new_z, ax=ax)
pl.show()




