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
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as pl
import scipy.io as sio
import pandas as pd

import pympedance.Synthesiser as imps
import pympedance.UNSW as pun
from pympedance.plot_utils import bodeplot
from scipy.io import matlab
from matplotlib.widgets import CheckButtons, SpanSelector


#pl.style.use('ggplot')

def read_impedance(filename):
    mm = matlab.loadmat(filename)
    zit = mm['Iteration'][0,-1]['Z']
    fvec=mm['Parameters'][0,0]['frequencyVector']
    harms = np.arange(mm['Parameters'][0,0]['harmLo'],
                      mm['Parameters'][0,0]['harmHi']+1)
    sout=mm['Iteration'][0,-1]['Output'][0,0]['spectrum'][harms,0]
    sr = mm['Parameters'][0,0]['samplingFreq'][0,0]
    nwind = mm['Parameters'][0,0]['numPoints'][0,0]
    tvec = (0.5+np.arange(zit.shape[1]))*nwind/sr
    specs = mm['Iteration'][0,0]['Input'][0,0]['totalSpectrum'].squeeze()
    n_chans = len(mm['Parameters'][0,0]['micSpacing'].squeeze())
    specs = specs[:,:,:n_chans]
    a = mm['Parameters'][0,0]['A']
    return {'TimeFreqImp': zit,
               'FreqVec': fvec,
               'TimeVec': tvec,
               'Specs':specs,
               'CalibMx':a,
               'Harms':harms}


def make_report(imp_obj):
    rep = ImVecWind(imp_obj)


class ImVecWind(object):
    def __init__(self,imp_obj):
        self.fig = pl.figure()
        self.tfmx = imp_obj['TimeFreqImp']
        self.fvec = imp_obj['FreqVec']
        self.tvec = imp_obj['TimeVec']
        self.specs = imp_obj['Specs']
        print(self.specs.shape)
        self.calibmx = imp_obj['CalibMx']
        self.harms = imp_obj['Harms']
        self.mode='mag'
        self.keep_prev = False
        self.slice_times = []

        self.ax_imped = [pl.subplot2grid((4,8), (2,0), colspan=7),
                    pl.subplot2grid((4,8),(3,0), colspan=7)]
        self.ax_tf = pl.subplot2grid((4,8), (0,0), rowspan=2,colspan=7)
                                        
        self.ax_cbar = pl.subplot2grid((4,12),(0,11),colspan=1,rowspan=2)

        
        # Make checkbuttons with all plotted lines with correct visibility
        rax = pl.subplot2grid((4,12),(2,11),rowspan=2,colspan=2)
        check = CheckButtons(rax, ['keep'], [False])
        check.on_clicked(self.checkboxes)

        self.ax_tf.imshow(20*np.log10(np.abs(self.tfmx)),
          extent=[0,self.tvec[-1],self.fvec[0],self.fvec[-1]],
          aspect='auto',
          origin='bottom',cmap='gray')
        pl.colorbar(cax=self.ax_cbar, mappable=self.ax_tf.images[0])

        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.mpl_connect('key_press_event', self.onpress)
        self.temp_lines=[]

        self.last_tidx = 0

        self.selector = SpanSelector(self.ax_tf, 
                                     self.onselect, 
                                     'horizontal', 
                                     useblit=True,
                                     rectprops=dict(alpha=0.5, 
                                                    facecolor='red'))

        self.spans = []
        pl.show()

    def checkboxes(self,label):
        if label == 'keep':
            self.keep_prev=not self.keep_prev

    def onpress(self, event):
        print(event.key)
        if event.key=='d' or event.key=='D':
            self.slice_times = []
            self.update()
            return

        if event.key=='left':
            self.last_tidx-=1
        if event.key=='right':
            self.last_tidx+=1

        if self.last_tidx <0:
            self.last_tidx=0
        if self.last_tidx>len(self.tvec)-1:
            self.last_tidx = len(self.tvec)-1

        try:
            self.slice_times[-1] = self.tvec[self.last_tidx]
        except IndexError:
            self.slice_times = [self.tvec[self.last_tidx]]

        self.update()

    def onselect(self, tmin, tmax):
        print(tmin,tmax)
        if not self.keep_prev:
            self.slice_times = [(tmin,tmax)]
        else:
            self.slice_times.append((tmin,tmax))
        self.update()

    def onclick(self, event):
        ax = event.inaxes

        # if ax is self.ax_tf:
        #     freq = event.ydata
        #     time = event.xdata
        #     print(time,freq)

        #     if not self.keep_prev:
        #         self.slice_times = [(time,time)]
        #     else:
        #         self.slice_times.append((time,time))

        #     self.update()

    def update(self):
        while 1:
            try:
                l = self.spans.pop(0)
            except IndexError:
                break
            l.remove()
            del l

        for axi in self.ax_imped:
            axi.cla()

        for time in self.slice_times:
            tidx = [np.argmin(np.abs(self.tvec-tt)) for tt in time]
            vvec = self.calc_mean_impedance(tidx[0],tidx[1])
            print(tidx)
            
            bodeplot(self.fvec, vvec, ax=self.ax_imped, lw=1)
            color = self.ax_imped[-1].get_lines()[-1].get_color()
            sp = self.ax_tf.axvspan(time[0],time[1], color=color,alpha=.3)
            self.spans.append(sp)
            self.last_tidx = tidx

        self.fig.canvas.draw()

    def calc_mean_impedance(self,imin,imax):
        specs = np.mean(self.specs[:,imin:imax,:],axis=1)
        pu=[]
        A = self.calibmx
        hlo = min(self.harms)
        hhi = max(self.harms)
        for fi,oi in enumerate(range(hlo,hhi+1)): 
            #ainv = pun.lscov(pp.A[:,:,fi])
            pui,_,_,_ = pun.lscov(A[:,:,fi],
                                specs[oi,:],
                                np.eye(2),
                                rcond=-1)
            pu.append(pui)
        pu = np.array(pu)
        return pu[:,0]/pu[:,1]



def parse_arguments(args):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('infile', help="Input file", default="",
                         nargs="?")



    return parser.parse_args(args)

if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    print(args.__dict__)

    if args.infile:
        imp_obj = read_impedance(args.infile)
    else:
        sys.exit(1)

    make_report(imp_obj)


