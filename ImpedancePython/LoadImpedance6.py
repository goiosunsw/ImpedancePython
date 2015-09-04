## Functionality to load impedance files in mat format
## saved by Impedance6 measurement program

import scipy.io as sio
import pylab as pl
import numpy as np
import PeakFinder as pf
#import AcousticResonator as ar

def GetImpedance6(mm, paramfile=None, freqLo=0.0, freqIncr=1.0):
    '''
    f, z = GetImpedance6(mm, paramfile=None, freqLo=0.0, freqIncr=1.0)
    
    Loads an Impedance 6 matlab structure.
    Parameter file is supplied in keyword argument paramfile
    or alternatively, supply freqLo and freqIncr to build 
    freqency vector
    '''
    
    data = mm
    z = data['iteration'][-1]['Z'][-1]
    
    if paramfile is not None:
        pp = sio.loadmat(paramfile)
        freqLo = pp['PARAMS']['freqLo'][0,0][0,0]
        freqHi = pp['PARAMS']['freqHi'][0,0][0,0]
        freqIncr = pp['PARAMS']['freqIncr'][0,0][0,0]
        #f = np.arange(freqLo,freqHi,freqIncr)
    
    npoints = len(z)
    f = freqLo + np.arange(npoints)*freqIncr
        
    return f,z

def GetImpedance7(mm):
    '''
    f, z = GetImpedance7(mm)
    mm = matlab structure
    Loads an Impedance 7 file in matlab format.
    '''
    
    
    # Read the mathfile
    #mm = sio.loadmat(filename)
    
    # The vector of interest is now called meanZ equivalent to Z in version 6
    z = mm['Iteration'][0,0]['meanZ']
    
    flo=mm['Parameters']['freqLo'][0,0][0,0]
    fhi=mm['Parameters']['freqHi'][0,0][0,0]
    
    npoints = len(z)
    f = np.linspace(flo,fhi,npoints)
        
    return f,z

def LoadImpedance(filename, paramfile=None, freqLo=0.0, freqIncr=1.0):
    '''
    Load an Impedance file, guessing version.
    Keyword arguments are supplied for version 6:
    * paramfile
    * freqLo and freqIncr to build freqency vector
    '''
    # Read the mathfile
    mm = sio.loadmat(filename)
    
    if 'Parameters' in mm.keys():
        f,z = GetImpedance7(mm)
    else:
        f,z = GetImpedance6(mm,paramfile,freqLo,freqIncr)
    
    return f,z

def PlotImpedance(f,z):
    pl.figure();

    ax1 = pl.subplot(2,1,1)
    pl.plot(f,20*np.log10(np.abs(z)), axes=ax1)
    pl.ylabel('Module (dB)')
    
    ax2 = pl.subplot(2,1,2,sharex=ax1)
    pl.plot(f,(np.angle(z)), axes=ax2)
    pl.ylabel('Phase (rad)')
    pl.xlabel('Frequency (Hz)')
    
    #legend(names)
    pl.show()
    
def Peaks(f,z):
    import PeakFinder as pk
    
    pf=pk.PeakFinder(abs(z))
    pf.refine_all(rad=3,logarithmic=True)
    pf.filter_by_salience(rad=5)
    
    fpk = np.interp(pf.get_pos(),np.arange(len(f)),f)
    zpk = np.interp(pf.get_pos(),np.arange(len(f)),abs(z.squeeze()))

    return fpk,zpk
    
    

