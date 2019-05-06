#!/usr/bin/env python
"""
    generate_broadband.py
    ---------------------

    Generates a flat spectrum broadband excitation signal with frequencies 
    limitted to a frequency band

    Depends on pympedance: https://github.com/goiosunsw/ImpedancePython.git

    Author: Andre Almeida <a.almeida@unsw.edu.au> 
    Creation date: 2019/03/20
    Version: 0.0.1
"""
import sys
import argparse
import numpy as np
from scipy.io import wavfile
from pympedance.UNSW import BroadBandExcitation

def parse_arguments(args):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-l', '--loop', type=int, default=1024,
                        help='number of samples per loop')
    parser.add_argument('-n', '--num-loops', type=int, default=8,
                        help='number of loops in track')
    parser.add_argument('-f', '--freq-lims', nargs=2, default=[0,np.inf],
                        help='minimum and maximum frequency of excitation')
    parser.add_argument('-s', '--sampling-rate', type=int, default=44100,
                        help='sampling rate')

    parser.add_argument('outfile', help="Output file", default="",
                         nargs="?")

    return parser.parse_args(args)


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    sr = args.sampling_rate
    fmax = float(args.freq_lims[1])
    if fmax > sr:
        fmax = sr

    fmin = float(args.freq_lims[0])
    if fmin<0:
        fmin = 0
    if fmin > fmax:
        fmin = 0
    
    print([fmin,fmax])

    bb=BroadBandExcitation(n_points=args.loop, 
                           n_cycles=args.num_loops, 
                           freq_lo=fmin, 
                           freq_hi=fmax,
                           sr=sr)
    w = bb.generate_sound() 
    w /= np.max(np.abs(w))*1.1
    wavfile.write(args.outfile, int(sr), w)

