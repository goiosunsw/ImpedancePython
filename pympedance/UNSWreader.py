# -*- coding: utf-8 -*-
"""
    ImpedanceUNSW
    ~~~~~~~~~~~~~~~~~~~

    Functionality to load impedance files in mat format
    saved by Impedance6 measurement program

    :copyright: (c) 2017 by Andre Almeida.
    :license: GPL, see LICENSE for more details.
"""

import string
import sys
import os
import numpy as np
from scipy.io import matlab
from Impedance import Impedance


def read_UNSW_impedance(filename=None, paramfile=None,
                        freqLo=None, freqIncr=1.0):
    """
    read a UNSW impedance measurement file and return an impedance object
    """
    imp_meas = ImpedanceMeasurement(filename=filename,
                                    paramfile=paramfile,
                                    freqLo=freqLo,
                                    freqIncr=freqIncr)

    imp_obj = Impedance(freq=imp_meas.f, imped=imp_meas.z)
    imp_obj.set_name(imp_meas.name)

    return imp_obj


class ImpedanceMeasurement(object):
    """
    Implements an Impedance that can read matlab files measured in UNSW
    and do a range of operations on them

    (obsolete: use read_UNSW_impedance())
    """

    def __init__(self, filename=None, paramfile=None,
                 freqLo=None, freqIncr=1.0):
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
                        self.buildFreqVect(freqLo, freqIncr)
                    else:
                        self.buildFreqVect(0.0, freqIncr)

            elif format == 'v7':
                self.readImpedance(filename)
            else:
                raise IOError('Unrecognised format for the impedance file')
        else:
            raise IOError("File not found:'%s", filename)

        # Keep track of derived impedances
        self.dernum = 0

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

        # The vector of interest is now called meanZ 
        # equivalent to Z in version 6
        z = mm['Iteration'][0, 0]['meanZ']

        flo = mm['Parameters']['freqLo'][0, 0][0, 0]
        fhi = mm['Parameters']['freqHi'][0, 0][0, 0]

        npoints = len(z)
        f = np.linspace(flo, fhi, npoints)

        self.f = f
        self.z = z

    def readImpedance6(self, filename):
        """
        Loads an Impedance 6 matlab structure.
        Parameter file is supplied in keyword argument paramfile
        or alternatively, supply freqLo and freqIncr to build 
        freqency vector
        """
        data = matlab.loadmat(filename)

        try:
            z = data['iteration'][-1]['Z'][-1]
        except ValueError:
            z = data['iteration']['analysis'][-1, -1]['Z'][-1]

        self.z = z

    def buildFreqVect(self, freqLo=0.0, freqIncr=1.0):
        npoints = len(self.z)
        self.f = freqLo + np.arange(npoints)*freqIncr

    def lookForParams(self, filename):
        """
        Try to find the param file.
        Usually they're in the parent directory or the same
        """

        fullpath = os.path.abspath(filename)
        curdir = os.path.dirname(fullpath)
        parentdir = os.path.abspath(os.path.join(curdir, os.pardir))

        try:
            dirlist = os.listdir(curdir)
            fidx = [xx.find('param') != -1
                    for xx in dirlist].index(True)
            return os.path.join(curdir, dirlist[fidx])
        except ValueError:
            try:
                dirlist = os.listdir(parentdir)
                fidx = [xx.find('param') != -1
                        for xx in dirlist].index(True)
                return os.path.join(parentdir, dirlist[fidx])
            except ValueError:
                sys.stderr.write('Could not find a parameter file')
                return None

    def readParams(self, paramfile):
        pp = matlab.loadmat(paramfile)
        try:
            ppp = pp['PARAMS']
        except KeyError:
            ppp = pp
        freqLo = np.squeeze(ppp['freqLo'])
        freqHi = np.squeeze(ppp['freqHi'])
        freqIncr = np.squeeze(ppp['freqIncr'])

        self.buildFreqVect(freqLo, freqIncr)

def lscov(a, b, w):
    """
    calculates the weighted least squared solution to
      a.x = b
    given the weights w
    """
    w = np.sqrt(np.diag(w))
    Aw = np.dot(w,a)
    Bw = np.dot(b,w)
    return np.linalg.lstsq(Aw, Bw)

def analyseinput(Input, Parameters, measType, countLoops=0):
    """
    analyseinput takes the spectra and determines pressure and flow (u) to
    save in Analysis structure
    """

    noiseCalculated = Input['numCycles'] >= 8
    #  calculate A and b matrices
    A = Parameters['A'].squeeze();
    harmLo = Parameters['harmLo'].squeeze()
    harmHi = Parameters['harmHi'].squeeze()
    nChannelFirst = Parameters['nChannelFirst'].squeeze()
    micSpacing = Paramters['micSpacing'].squeeze()
    nMics = len(micSpacing)
    fVec = Parameters['frequencyVector'].squeeze()



    meanSpectrum = Input['meanSpectrum'].squeeze()
    totalSpectrum = Input['totalSpectrum'].squeeze()
    spectralError = Input['spectralError'].squeeze()

    spectralError = spectralError[harmLo:harmHi, nChannelFirst:nMics]

    if noiseCalculated: 
        # Fit a function of the form Af^n to the noise data,
        # and replace the noise data with the (smooth) function.
        fitData = [np.ones(fVec.shape), np.log(fVec)]
        coeff = np.linalg.lstsq(fitData,np.log(spectralError))
        spectralError = np.exp(fitData*coeff)
        sigmaVariable = np.transpose(spectralError, axes=[1,2,0])
    else:
        # otherwise, choose a noise function of f^(-0.5)
        sigmaVariable = (fVec**(-0.5) *
                         np.ones((1,spectralError.shape[1])))
        sigmaVariable = np.transpose(sigmaVariable, axes=[1,2,0])

    if measType == 'Averaged':
        b = np.transpose(meanSpectrum[harmLo:harmHi,nChannelFirst:nMics],
            axes=[1,2,0])
    elif measType == 'Individual':
        if countLoops:
            # reorder the totalSpectrum vector to match meanSpectrum
            if nMics == 1:
                tempSpectrum =\
                    totalSpectrum[harmLo:harmHi, countLoops-1, 0];
            elif nMics == 2:
                tempSpectrum =\
                    [totalSpectrum[harmLo: harmHi, countLoops-1, 0],\
                    totalSpectrum[harmLo: harmHi, countLoops-1, 1]]
            else:
                tempSpectrum =\
                    [totalSpectrum[harmLo: harmHi, countLoops-1, 0],\
                    totalSpectrum[harmLo: harmHi, countLoops-1, 1],\
                    totalSpectrum[harmLo: harmHi, countLoops-1, 2]]
      
        b = np.transpose(tempSpectrum, axes=[1,2,0])

    # initialise p, u, deltap and deltau
    p = np.zeros(A.shape[2],1)
    deltap = np.zeros(p.shape)
    u = np.zeros(p.shape)
    deltau = np.zeros(p.shape)
    for freqCount in range(A.shape[2]): # length of frequency vector
        # Calculate the covariant matrix by putting the elements of
        # sigmaVariable along the diagonal
        covar = np.diag(sigmaVariable[:,:,freqCount]**2)
        # if only two mics, calculate x using backslash operator
        # i.e. solve A*x = B for x
        if nMics == 1:
            x = np.linalg.lstsq(A[:,:,freqCount],b[:,:,freqCount])
            # matrix is not square so use pseudoinverse
            Ainv = np.linalg.pinv(A[:,:,freqCount])
        elif nMics == 2:
            x = np.linalg.lstsq(A[:,:,freqCount],b[:,:,freqCount])
            Ainv = np.linalg.inv(A[:,:,freqCount])
        else:
            # otherwise, use a weighted least-squares to determine x
            # weighted least squares solution to A*x = b with weighting covar
            x = lscov(A[:,:,freqCount], b[:,:,freqCount], covar)
            Ainv = np.linalg.pinv(A[:,:,freqCount])
        # Use the inverse (or pseudoinverse) to calculate dx
        dx = np.sqrt(np.diag(Ainv*covar*Ainv.T))
        # Calculate the impedance and its error
        Z0 = Parameters.Z0
        p[freqCount] = x[0]
        u[freqCount] = x[1] / Z0
        deltap[freqCount] = dx[0]
        deltau[freqCount] = dx[1] / Z0

    return dict(p=p, u=u, deltap=deltap, deltau=deltau, z0=Z0)
