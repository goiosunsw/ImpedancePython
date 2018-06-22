# coding: utf-8

# # Impedance benchmarks
# The aim is to compare ipedances of ducts under different conditions,
# against modelled impedances from the known geometries

import numpy as np
import matplotlib.pyplot as pl
import sys
import os
import glob
import scipy.io as sio
import argparse as ap
import logging

#import pdb
#pdb.set_trace()

import pympedance.UNSW as pun

import pympedance.Synthesiser as psy
from pympedance.plot_utils import bodeplot


def expand_filelists(args):
    # expand profiles
    if not args.profile:
        if args.name:

            globstr = os.path.join(args.profile_dir,args.name+'*.csv')
            profile_list = glob.glob(globstr)
        else:
            globstr = os.path.join(args.profile_dir,'*.csv')
            profile_list = glob.glob(globstr)
    else:
        profile_list = [args.profile]

    if not args.impedance:
        globstr = os.path.join(args.impedance_dir,'*.mat')
        impedance_list = glob.glob(globstr)
    else:
        impedance_list = [args.impedance]

    file_dict = {}
    for profile_file in profile_list:
        base_name, ext = os.path.splitext(os.path.split(profile_file)[-1])
        fl = []
        logging.info(base_name)
        for imp_file in impedance_list:
            if base_name in imp_file:
                imp_trunk,_ = os.path.splitext(imp_file)
                bn_start = imp_trunk.find(base_name)
                trail = imp_file[bn_start+len(base_name):]
                args = trail.split('_')
                fl.append((imp_file,*args))

                logging.info('    '+imp_file+', '.join(args))
        file_dict[base_name] = fl
        return file_dict

def process_files(file_dict, profile_path='.'):
    for profile, imp_list in file_dict.items():
        profile_file = os.path.join(profile_path,profile+'.csv')
        vt_raw = psy.vocal_tract_reader(profile_file, unit_multiplier=0.001)
        fig,ax = pl.subplots(2,sharex=True)
        ax[0].set_title(profile)
        for imp_file, *args in imp_list:
            io = pun.ImpedanceMeasurement(imp_file)
            bodeplot(io.f,io.z,ax=ax)
            vt = vt_raw.copy()
            if 'Closed' in args or 'closed' in args:
                vt.set_termination(psy.PerfectClosedEnd())
            elif 'Open' in args or 'open' in args:
                vt.set_termination(psy.FlangedPiston(radius=vt.elements[-1].output_radius))

            vt.plot_impedance(ax=ax)
    pl.legend()
    pl.show()

def main(args):
    if args.iterate:
        file_dict = expand_filelists(args)
    process_files(file_dict, profile_path = args.profile_dir)
    #logging.info('Nothing to do')
    return 0

def parse_args(arglist):
    parser = ap.ArgumentParser()

    parser.add_argument('name', nargs='?', help='measurement name')

    parser.add_argument('-b', '--base-dir', help='base dir')
    parser.add_argument('-p', '--profile', help='profile file')
    parser.add_argument('-i', '--impedance', help='impedance')
    parser.add_argument('-P', '--profile-dir', help='profile directory')
    parser.add_argument('-I', '--impedance-dir', help='impedance directory')
    parser.add_argument('-l', '--log', help='log level', default='info')
    args = parser.parse_args(arglist)
    args.iterate=True
    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)
    logging.basicConfig(level=numeric_level)
        
    if args.profile and args.impedance:
        args.iterate = False
   
    if not args.base_dir:
        basedir = os.getcwd()
    else:
        basedir = os.path.abspath(os.path.expanduser(args.base_dir))
    args.base_dir = basedir

    if not args.profile_dir:
        args.profile_dir = os.path.join(basedir, 'profiles')
    args.profile_dir = os.path.abspath(os.path.expanduser(args.profile_dir))

    if not args.impedance_dir:
        args.impedance_dir = os.path.join(basedir, 'impedance')
    args.impedance_dir = os.path.abspath(os.path.expanduser(args.impedance_dir))


    logging.info('Profile dir: '+args.profile_dir)
    logging.info('Impedance dir: '+args.impedance_dir)

    logging.info('Profile file: '+str(args.profile))
    logging.info('Impedance file: '+str(args.impedance))

    logging.info('Data name: '+str(args.name))

    return args

if __name__ == "__main__":
    args=parse_args(sys.argv[1:])
    sys.exit(main(args))
