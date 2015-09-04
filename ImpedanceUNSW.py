## Functionality to load impedance files in mat format
## saved by Impedance6 measurement program

#import scipy.io as sio
from scipy.io import matlab
import pylab as pl
import numpy as np
import string
import PeakFinder as pf
import os

class ImpedanceMeasurement(object):
    '''
    Implements an Impedance that can read matlab files measred in UNSW
    and do a range of operations on them
    '''
    def __init__(self, filename = None, paramfile = None, freqLo=None, freqIncr=1.0):
        if filename is not None:
            fileext = os.path.basename(filename)
            base = os.path.splitext(fileext)
            self.name = os.path.basename(base[0])
            format = self.detectFormat(filename)
            if format == 'v6':
                self.readImpedance6(filename)
                if paramfile is None:
                    paramfile = self.lookForParams(filename)
                
                if paramfile is not None:
                    self.readParams(paramfile)
                else:
                    if freqLo is not None:
                        self.buildFreqVect(freqLo,freqIncr)
                    else:
                        self.buildFreqVect(0.0,freqIncr)
                
                
            elif format == 'v7':
                self.readImpedance(filename)
            else: 
                raise IOError('Unrecognised format for the impedance file')
        else:
            raise IOError('File not found:''%s''',filename)
    
    def set_name(self,name):
        self.name=name
    
    def detectFormat(self, filename):
        format = ''
        struct = matlab.whosmat(filename)
        fields = [ss[0] for ss in struct]
        if 'Parameters' in fields:
            format = 'v7'
        
        if 'calibFile' in fields:
            format = 'v6'
            
        return format
    
    def readImpedance(self, filename):
        # Read the mathfile
        mm = matlab.loadmat(filename)
        
        # The vector of interest is now called meanZ equivalent to Z in version 6
        z = mm['Iteration'][0,0]['meanZ']
    
        flo=mm['Parameters']['freqLo'][0,0][0,0]
        fhi=mm['Parameters']['freqHi'][0,0][0,0]
    
        npoints = len(z)
        f = np.linspace(flo,fhi,npoints)
        
        self.f = f
        self.z = z
        
    def readImpedance6(self, filename):
        '''
        Loads an Impedance 6 matlab structure.
        Parameter file is supplied in keyword argument paramfile
        or alternatively, supply freqLo and freqIncr to build 
        freqency vector
        '''
        data = matlab.loadmat(filename)
        
        z = data['iteration'][-1]['Z'][-1]
        
        self.z = z
    
    def buildFreqVect(self, freqLo=0.0, freqIncr=1.0):
        npoints = len(self.z)
        self.f = freqLo + np.arange(npoints)*freqIncr
        
    def lookForParams(self, filename):
        '''
        Try to find the param file.
        Usually they're in the parent directory or the same
        '''
        
        fullpath = os.path.abspath(filename)
        curdir = os.path.dirname(fullpath)
        parentdir = os.path.abspath(os.path.join(curdir, os.pardir))
        
        try:
            dirlist = os.listdir(parentdir)
            fidx = [string.find(xx,'param')!=-1 for xx in dirlist].index(True)
            return os.path.join(parentdir,dirlist[fidx])
        except ValueError:
            try:
                dirlist = os.listdir(parentdir)
                fidx = [string.find(xx,'param')!=-1 for xx in dirlist].index(True)
                return os.path.join(parentdir,dirlist[fidx])
            except ValueError:
                sys.stderr.write('Could not find a parameter file')
                return None
        
    def readParams(self,paramfile):
        pp = matlab.loadmat(paramfile)
        freqLo = pp['PARAMS']['freqLo'][0,0][0,0]
        freqHi = pp['PARAMS']['freqHi'][0,0][0,0]
        freqIncr = pp['PARAMS']['freqIncr'][0,0][0,0]
        
        self.buildFreqVect(freqLo,freqIncr)
        
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
        
    def findPeaks(self):
        import PeakFinder as pk
        
        f = self.f
        pf=pk.PeakFinder(abs(self.z))
        pf.refine_all(rad=3,logarithmic=True)
        pf.filter_by_salience(rad=5)
    
        fpk = np.interp(pf.get_pos(),np.arange(len(f)),f)
        zpk = np.interp(pf.get_pos(),np.arange(len(f)),abs(self.z.squeeze()))

        return fpk,zpk
    
        
    
        