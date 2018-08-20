import pympedance.UNSW as pun
import numpy as np
import sys

io = pun.ImpedanceMeasurement(sys.argv[1])
par = io.parameters
mic_vec = np.tile(io.mean_spectrum[10,:],(int(par.num_points/2)+1,1))
par.analyse_input(mic_vec)
