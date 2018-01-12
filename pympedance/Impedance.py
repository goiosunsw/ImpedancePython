# -*- coding: utf-8 -*-
"""
    Impedance
    ~~~~~~~~~

    Basic impedance object with plotting and other functionalities

    :copyright: (c) 2017 by Andre Almeida (UNSW).
    :license: GPL, see LICENSE for more details.
"""


#import scipy.io as sio
import pylab as pl
import numpy as np
import string
from . import PeakFinder as pf
import os


class Impedance(object):
    """
    Impedance object containing impedance at a set of frequencies
    """
    
    def __init__(self, freq=None, imped=None):
        if imped is not None:
            self.imped = imped
        else:
            self.imped = np.array([])

        if freq is not None:
            assert len(freq) == len(self.imped), \
                    "Frequency (%s) and impedance (%s) are not same size" % (
                        freq.shape, self.imped.shape)
            self.freq = freq
        else:
            self.freq = np.linspace(0,0.5,len(self.imped))

    def set_name(self,name):
        self.name=name
        
    def get_name(self):
        return self.name
    
    def add_suffix_to_name(self, suf='Corr'):
        cur = self.get_name()
        sufst = str.find(cur,'-'+suf)
        if self.dernum > 0:
            basename = cur[0:sufst]
            remaining = cur[sufst+len(suf)+2:]
            n = self.dernum
            newname = (cur[0:sufst] + '-%s-%d')%(suf, n)
        else:
            newname = (cur + '-%s')%(suf)
        
        self.dernum += 1
        self.set_name(newname)
    
    def GuessNominalFreqFromFileName(self):
        '''tries to guess note frequency from the file name
        '''
        
        import aubio
        import re
        
        tokens=re.findall('([A-Ga-g][sS#bB+-]*\d+)',self.name)
    
        try:
            notename=re.sub('s','#',tokens[0])
            f=aubio.miditofreq(aubio.note2midi(notename))
        except (ValueError,IndexError):
            f=np.NaN
    
        return f
    
    def copy(self):
        '''Create a new object with the same values as this one'''
        import copy
        
        return copy.deepcopy(self)
    
    def correctImpedance(self, g=lambda x,y : x):
        ''' Apply a function g to correct the measured impedance
         the format of g is g(Zi,f), where Zi is the measured impedance
         and f the frequency'''
         
        newimp = self.copy()
        
        zraw = self.getImpedance()
        f = self.getFrequencyVect()
        
        newimp.z = g(zraw,f)
        newimp.add_suffix_to_name(suf='Corr')
        
        return newimp
        
    def addParallelMouthpiece(self, vol = 3.5e-7, mass = 4500.0, res = 0.0):
        ''' Calculate the "Impedance seen by the mouthpiece flow", 
        given the reed parameters:
          * vol:  equivalent volume of the reed (acoustic) 
          * mass: equivalent acoustic mass of the reed
          * res:  equivalent acoustic resistance
        '''
        
        # reed equivalent volume
        eqvol = vol
        gamma=1.4
        p0=1.013e5
        eqc = eqvol / ( gamma * p0 )
        # reed equivalent resistance
        eqr = res
        # reed acoustic mass
        eqm = mass
    
        # reed impedance
        zreed = lambda ff: 1./(1j*2.*np.pi*ff*eqc) + eqr + 1j*2.*np.pi*ff*eqm
        g = lambda zi,ff: 1./zi + 1./zreed(ff)
        
        newimp = self.correctImpedance()
        
        return newimp.getImpedance()
        
    def getImpedance(self):
        return self.z.squeeze()

    def getFrequencyVect(self):
        return self.f

        
    def findPeaks(self):
        '''Finds the frequencies and values of impedance maxima
        '''
        import PeakFinder as pk
        
        f = self.f
        pf=pk.PeakFinder(abs(self.z))
        pf.refine_all(rad=3,logarithmic=True)
        pf.filter_by_salience(rad=5)
    
        fpk = np.interp(pf.get_pos(),np.arange(len(f)),f)
        zpk = np.interp(pf.get_pos(),np.arange(len(f)),abs(self.z.squeeze()))

        return fpk,zpk

    def findZeroPh(self, direction=-1):
        '''Finds the frequencies at which the phase is 0
        * default, find zero crossings with negative slope
        * direction=+1 finds those with positive slope
        '''
        
        f = self.getFrequencyVect()
        z = self.getImpedance()
        za = np.angle(z)*direction
        
        zci=np.nonzero(np.all((za[:-1]<0.,za[1:]>0.),axis=0))
        azcf = []
        for ii in zci:
            azcf.append((f[ii]-(f[ii+1]-f[ii])/(za[ii+1]-za[ii])*za[ii]))
        
        zcf = np.array(azcf).squeeze()

        return zcf,np.interp(zcf,f,np.abs(z))

    
    def findPeaksCorrected(self, vol = 3.5e-7, mass = 4500.0, res = 0.0):
        import PeakFinder as pk
        
        f = self.f
        z = self.addParallelMouthpiece(vol=vol,mass=mass,res=res)
        pf=pk.PeakFinder(abs(z))
        pf.refine_all(rad=3,logarithmic=True)
        pf.filter_by_salience(rad=5)
    
        fpk = np.interp(pf.get_pos(),np.arange(len(f)),f)
        zpk = np.interp(pf.get_pos(),np.arange(len(f)),abs(z))

        return fpk,zpk
 
    
    def plotAbsAngle(self,fig=None):
        if fig is None:
            fig=pl.figure()
        else:
            if type(fig) is int:
                fig = pl.figure(fig)
            else:
                fig=pl.figure(fig.number)
        ax1=pl.subplot(211)
        pl.hold(True)
        pl.plot(self.f,20*np.log10(np.abs(self.z)),label=self.name)
        pl.xlabel('Frequency (Hz)')
        pl.ylabel('Module (dB)')
        
        pl.subplot(212,sharex=ax1)
        pl.hold(True)
        pl.plot(self.f,(np.angle(self.z)),label=self.name)
        pl.xlabel('Frequency (Hz)')
        pl.ylabel('Argument (rad)')
        
        pl.legend()
        
        return fig

