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
parser.add_argument('-P', '--no-pipe', action='store_true', 
                    help='disable infinite pipe calibration')
parser.add_argument('-F', '--no-flange', action='store_true', 
                    help='disable infinite flange calibration')
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

dict_convert = {'inf_imp': 'infinite_imp_file',
                'inf_pipe': 'infinite_pipe_file',
                'inf_flange': 'infinite_flange_file'}

# Read the impedance file with its parameters
io = ImpedanceMeasurement(meas_file)

#calibs = io.parameters.find_calib_files()
calibs = io.parameters.calib_files

calib_args = dict()
for k,v in dict_convert.items():
    try:
        calib_args[v] = calibs[k]
    except KeyError:
        calib_args[v] = None



# The calibration files corresponding to the above measurement
if args.impedance:
    calib_args['infinite_imp_file'] = args.impedance
if args.pipe:
    calib_args['infinite_pipe_file'] = args.pipe
if args.flange:
    calib_args['infinite_flange_file'] = args.flange

if args.no_pipe:
    calib_args.pop('infinite_pipe_file',None) 
if args.no_flange:
    calib_args.pop('infinite_flange_file',None) 

print('Calibration dict:')
for k,v in calib_args.items():
    print('  '+k+': '+v)

# Plot it
fig,ax = bodeplot(io.f,io.z)

# Recalculate the calibration matrix
old_a = io.parameters.A
new_a = io.parameters.calc_calibration_matrix(**calib_args)

# Set the matrix in parameters
# (will probably be done automatically in a later version)
io.parameters.A = new_a
# recalculate the impedance with the new calibration
new_z = io.calculate_impedance()

# Plot the new impedance on top of the first one
bodeplot(io.f,new_z, ax=ax)

# Set the matrix in parameters
# (will probably be done automatically in a later version)
#io.parameters.A = old_a
# recalculate the impedance with the new calibration
#new_z = io.calculate_impedance()

# Plot the new impedance on top of the first one
#bodeplot(io.f,new_z, ax=ax)
pl.show()




