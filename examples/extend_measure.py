import numpy as np
import matplotlib.pyplot as pl
import sys
import os
import scipy.io as sio

from pympedance.UNSW import *
from pympedance.Synthesiser import *
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
meas_file = '/home/goios/Data/20180301/meas/17cmPipe.mat'

# Read the impedance file with its parameters
io = ImpedanceMeasurement(meas_file)

# Plot it
fig,ax = bodeplot(io.f,io.z,label='meas')
# extend the measurement with a short cylinder
l_ext = .01
r = .026/2
duct = Duct()
duct.append_element(StraightDuct(length=l_ext, radius=r))
term = io.as_interpolated_impedance()
duct.set_termination(term)

# Plot it
fig,ax = bodeplot(io.f,duct.get_input_impedance_at_freq(io.f),label='meas_ext',ax=ax)



# build a cylinder model to compare
l = .17
duct = Duct()
duct.append_element(StraightDuct(length=l, radius=r))
duct.set_termination(FlangedPiston(radius=r))

# Plot it
fig,ax = bodeplot(io.f,duct.get_input_impedance_at_freq(io.f),label='model',ax=ax)

# export interpolated impedance
cyl_int = duct.as_interpolated_impedance(f=io.f)


# extend the measurement with a short cylinder
duct = Duct()
duct.append_element(StraightDuct(length=l_ext, radius=r))
term = cyl_int
duct.set_termination(term)

# Plot it
fig,ax = bodeplot(io.f,duct.get_input_impedance_at_freq(io.f),label='model_ext',ax=ax)



pl.legend()
pl.show()




