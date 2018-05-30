import numpy as np
import matplotlib.pyplot as pl
import sys
import os
import scipy.io as sio
import argparse as ap

from pympedance.UNSW import *
from pympedance.plot_utils import bodeplot

parser = ap.ArgumentParser()

parser.add_argument('filenames', nargs='*', help='measurement filename')
parser.add_argument('-o', '--output', nargs='?', help='output to file')
args = parser.parse_args()

#import pdb
#pdb.set_trace()

# Define path for the data
script_path, _ = os.path.split(os.path.realpath(__file__))
print(script_path)

# The actual impedance measurement file
if args.filenames:
    meas_files = args.filenames
else:
    meas_files = [os.path.join(script_path, '../tests/data/i.mat')]

fig,ax = pl.subplots(2,sharex=True)
for fn in meas_files:
    io = ImpedanceMeasurement(fn)

    name = os.path.splitext(os.path.basename(fn))[0]
    fig,ax = bodeplot(io.f,io.z,ax=ax,label=name)

pl.legend()
pl.show()




