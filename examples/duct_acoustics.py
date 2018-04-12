#!/usr/bin/env python
"""
    duct_acoustics
    --------------

    Calculates simulated impedance and pressure distribution in
    an acoustic duct.

    Duct geometry can be given (in meters) in a CSV file containing tow
    columns:
        * 1st column: segment length
        * 2nd column: segment radius

    Plots duct geometry, impedance and pressure distribution

    Depends on pympedance: https://github.com/goiosunsw/ImpedancePython.git

    Author: Andre Almeida <a.almeida@unsw.edu.au> 
    Creation date: 2018/04/02
    Version: 0.0.1
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as pl
import scipy.io as sio
import pandas as pd

import pympedance.Synthesiser as imps
from pympedance.plot_utils import bodeplot

pl.style.use('ggplot')

def profile_to_duct(l=None, rad=None, x=None, reverse=False, 
                    termination='piston', loss_multiplier=None):
    """
    Generate a duct object based on lists of cylindrical segments
    with lengths l and radii rad
    """
    vt = imps.Duct()

    if l is None:
        #l=np.concatenate([[0],np.diff(x)])
        l = np.diff(x)
        #rad = (rad[:-1]+rad[1:])/2
        
    if not reverse:
        l = np.flipud(l)
        rad = np.flipud(rad)
    
    for ll, rr in zip(l,rad):
        #ll = (xx-prevx)
        if ll>0:
            vt.append_element(imps.StraightDuct(length=ll,radius=rr,loss_multiplier=loss_multiplier))
            
    
    if termination=='piston':
        vt.set_termination(imps.FlangedPiston(radius=rr))
    elif termination=='open':
        vt.set_termination(imps.PerfectOpenEnd())
    elif termination=='closed':
        vt.set_termination(imps.PerfectClosedEnd())
        
    return vt

def vocal_tract_reader(filename, columns=None, 
                       skiprows=0,
                       unit_multiplier=1.):
    """
    reads a vocat tract file with two columns:
        * By default first column has lengths and second column has radii in m
        * Valid column names are:
            + lengths
            + positions (segment limits: first segment starts at 0)
            + radii
            + area
        * Choose unit multiplier to convert units to m. for example, 1000 means
        the units of the file is mm, and .5 means the units are in m but the
        column represents diameter

    returns a pympedance.Synthesiser.Duct object
    """
    if not columns:
        columns = ['len','rad']

    vtpd = pd.read_csv(filename, header=None, skiprows=skiprows)
    for ic, col in enumerate(columns):
        if col[:3] == 'len':
            l = np.array(vtpd[ic].tolist())
        elif col[:3] == 'pos':
            x = np.array(vtpd[ic].tolist())
            x = np.concatenate(([0], x))
            l = np.diff(x)
        elif col[:3] == 'rad':
            r = np.array(vtpd[ic].tolist())
        elif col[:4] == 'diam':
            r = np.array(vtpd[ic].tolist())/2
        elif col[:4] == 'area':
            r = (np.array(vtpd[ic].tolist())/np.pi)**.5
        else:
            sys.stderr.write('Column {} ({}) skipped\n'.format(ic, col))
        
    # find 0 or negative lengths, warn and remove them
    nnp = (l<=0)
    if any(nnp):
        sys.stderr.write('Segments skipped:\n')
        for ii,b in enumerate(nnp):
            if b:
                sys.stderr.write('  {}: l={}, r={}\n'.format(ii,l[ii],r[ii]))
        r = r[np.logical_not(nnp)]
        l = l[np.logical_not(nnp)]
        
    vt = profile_to_duct(l=l*unit_multiplier, rad=r*unit_multiplier,
                         termination='piston')
    return vt

def make_report(duct,fmax=4000):
    rep = ReportWind(duct=duct, fmax=fmax)


class ReportWind(object):

    def __init__(self,duct,fmax=4000):
        self.fig = pl.figure()
        self.duct=duct
        self.var='transpedance'
        self.ax_geom = pl.subplot2grid((4,4), (1,0), rowspan=3)
        self.ax_vbore = self.ax_geom.twiny()
        self.ax_imped = [pl.subplot2grid((4,4), (0,1), colspan=3),
                    pl.subplot2grid((4,4),(1,1), colspan=3)]
        self.ax_pdist = pl.subplot2grid((4,4), (1,1), rowspan=3,colspan=3,
                                        sharex=self.ax_imped[0],
                                        sharey=self.ax_geom)
        self.ax_cbar = pl.subplot2grid((4,12),(0,0))

        duct.plot_geometry(ax=self.ax_geom, vert=True)
        duct.plot_impedance(ax=self.ax_imped,npoints=1000,fmax=fmax)
        self.fvec = self.ax_imped[0].lines[-1].get_xdata()
        duct.plot_acoustic_distribution(ax=self.ax_pdist, fmax=fmax,
                                        var=self.var)
        pl.colorbar(cax=self.ax_cbar, mappable=self.ax_pdist.images[0])

        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.temp_lines=[]

        pl.show()

    def onclick(self,event):
        ax = event.inaxes
        unremoved_lines=[]

        if ax is self.ax_pdist:

            freq = event.xdata
            pos = event.ydata

            for ll in self.temp_lines:
                try:
                    ll.remove()
                    
                except AttributeError:
                    unremoved_lines.append(ll)

            self.temp_lines = unremoved_lines
            self.ax_vbore.relim()

            self.temp_lines.append(ax.axvline(freq,ls='--',color='k'))
            self.temp_lines.append(ax.axhline(pos,ls='--',color='k'))
            vvec = self.duct.var_transfer_func(to_pos=pos, freq=self.fvec, var=self.var)

            bodeplot(self.fvec,vvec,ax=self.ax_imped,lw=1,color='k')
            for axi in self.ax_imped:
                self.temp_lines.append(axi.get_lines()[-1])

            posvec = np.linspace(0,duct.get_total_length(),100)
            vdist = [self.duct.var_transfer_func(to_pos=pp,freq=np.array([freq]),
                                                var =self.var) for pp in
                     posvec]
            vdist=np.array(vdist).squeeze()
            ln = self.ax_vbore.plot(np.abs(vdist),posvec,lw=1,color='k')
            self.temp_lines.append(ln[0])

            pl.show()

def example_duct(rad0=.0075, rad_e=.0035, lb=.3, l0=.5):

    world = imps.AcousticWorld()
    duct = imps.Duct(world=world,losses=True)
    duct.append_element(imps.StraightDuct(length=lb,radius=rad0))
    #duct.append_element(imps.StraightDuct(length=l0-lb,radius=rad_e))
    lastrad = duct.elements[-1].radius
    duct.set_termination(imps.FlangedPiston(radius=lastrad))
    return duct

def parse_arguments(args):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('infile', help="Input file", default="",
                         nargs="?")
    parser.add_argument('-o', '--outfile', help="Output file",
                        default=sys.stdout, type=argparse.FileType('w'))
    parser.add_argument('-c', '--columns', help="CSV columns (position, length, radii, area)",
                        action='append',default=[])
    parser.add_argument('-m', '--multiplier', help="unit multilplier (.001 for mm)",
                        default=1., type=float)
    parser.add_argument('-f', '--fmax', help="maximum frequency",
                        default=4000., type=float)



    return parser.parse_args(args)

if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    print(args.__dict__)

    if args.infile:
        duct = vocal_tract_reader(args.infile,
                                  columns=args.columns,
                                  unit_multiplier=args.multiplier)
    else:
        duct = example_duct()

    make_report(duct, fmax=args.fmax)


