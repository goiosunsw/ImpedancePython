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
from ._impedance import Impedance
from copy import copy


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

def analyseinput(Input, Parameters, 
                 measType='Averaged', 
                 countLoops=0):
    """
    analyseinput takes the spectra and determines pressure and flow (u) to
    save in Analysis structure
    """

    numCycles = Input['numCycles'].squeeze()
    noiseCalculated = numCycles[0] >= 8
    #  calculate A and b matrices
    A = Parameters['A'].squeeze();
    harmLo = Parameters['harmLo'].squeeze()
    harmHi = Parameters['harmHi'].squeeze()
    nChannelFirst = Parameters['nChannelFirst'].squeeze()
    micSpacing = Parameters['micSpacing'].squeeze()
    nMics = len(micSpacing)
    # shouldn't it be...
    # nChannelLast = nChannelFirst + nMics
    nChannelLast = nMics
    fVec = Parameters['frequencyVector'].squeeze()



    meanSpectrum = Input['meanSpectrum'].squeeze()
    totalSpectrum = Input['totalSpectrum'].squeeze()
    spectralError = Input['spectralError'].squeeze()

    spectralError = spectralError[harmLo:harmHi+1, nChannelFirst-1:nChannelLast]

    if noiseCalculated: 
        # Fit a function of the form Af^n to the noise data,
        # and replace the noise data with the (smooth) function.
        fitData = np.array([np.ones(fVec.shape), np.log(fVec)]).T
        coeff,_,_,_ = np.linalg.lstsq(fitData,np.log(spectralError))
        spectralError = np.exp(np.dot(fitData, coeff))
        sigmaVariable = np.transpose(np.array([spectralError]),
                                     axes=[2,0,1])
    else:
        # otherwise, choose a noise function of f^(-0.5)
        sigmaVariable = (fVec**(-0.5) *
                         np.ones((1,spectralError.shape[1])))
        sigmaVariable = np.transpose(np.array([sigmaVariable]), 
                                     axes=[2,0,1])

    if measType == 'Averaged':
        meanSpecRed = meanSpectrum[harmLo:harmHi+1,nChannelFirst-1:nChannelLast]
        b = np.transpose(np.array([meanSpecRed]),
                         axes=[2,0,1])
    elif measType == 'Individual':
        if countLoops:
            # reorder the totalSpectrum vector to match meanSpectrum
            if nMics == 1:
                tempSpectrum =\
                    totalSpectrum[harmLo:harmHi+1, countLoops-1, 0];
            elif nMics == 2:
                tempSpectrum =\
                    [totalSpectrum[harmLo:harmHi+1, countLoops-1, 0],\
                     totalSpectrum[harmLo:harmHi+1, countLoops-1, 1]]
            else:
                tempSpectrum =\
                    [totalSpectrum[harmLo: harmHi+1, countLoops-1, 0],\
                    totalSpectrum[harmLo: harmHi+1, countLoops-1, 1],\
                    totalSpectrum[harmLo: harmHi+1, countLoops-1, 2]]
      
        b = np.transpose(tempSpectrum, axes=[2,0,1])

    # initialise p, u, deltap and deltau
    p = np.zeros((A.shape[2],1),dtype='complex')
    deltap = np.zeros_like(p)
    u = np.zeros_like(p)
    deltau = np.zeros_like(p)
    for freqCount in range(A.shape[2]): # length of frequency vector
        # Calculate the covariant matrix by putting the elements of
        # sigmaVariable along the diagonal
        covar = (sigmaVariable[:,0,freqCount]**2)
        # if only two mics, calculate x using backslash operator
        # i.e. solve A*x = B for x
        if nMics == 1:
            x,_,_,_ = np.linalg.lstsq(A[:,:,freqCount],b[:,:,freqCount])
            # matrix is not square so use pseudoinverse
            Ainv = np.linalg.pinv(A[:,:,freqCount])
        elif nMics == 2:
            x,_,_,_ = np.linalg.lstsq(A[:,:,freqCount],b[:,:,freqCount])
            Ainv = np.linalg.inv(A[:,:,freqCount])
        else:
            # otherwise, use a weighted least-squares to determine x
            # weighted least squares solution to A*x = b with weighting covar
            x,_,_,_ = lscov(A[:,:,freqCount], b[:,0,freqCount], covar)
            Ainv = np.linalg.pinv(A[:,:,freqCount])
        # Use the inverse (or pseudoinverse) to calculate dx
        dx = np.sqrt(np.diag(np.dot(np.dot(Ainv, np.diag(covar)),
                             Ainv.T)))
        # Calculate the impedance and its error
        Z0 = Parameters['Z0'].squeeze()
        p[freqCount] = x[0]
        u[freqCount] = x[1] / Z0
        deltap[freqCount] = dx[0]
        deltau[freqCount] = dx[1] / Z0

    return dict(p=p, u=u, deltap=deltap, deltau=deltau, z0=Z0)


def recalc_calibration(InfPipe=None, 
                       InfImp=None,
                       InfFlange=None,
                       Parameters=None):
    """
    recalculate calibration parameters from experimental calibration
    measurements
    """

    
    A_old = Parameters['A'].squeeze()
    A = copy(A_old)
    z0 = Parameters['Z0'].squeeze()
    harmLo = Parameters['harmLo'].squeeze()
    harmHi = Parameters['harmHi'].squeeze()
    nChannelFirst = Parameters['nChannelFirst'].squeeze()
    micSpacing = Parameters['micSpacing'].squeeze()
    nMics = len(micSpacing)
    # shouldn't it be...
    # nChannelLast = nChannelFirst + nMics
    nChannelLast = nMics
    fVec = Parameters['frequencyVector'].squeeze()

    knownZ = [np.inf*np.ones(A_old.shape[2]),
              z0 * np.ones(A_old.shape[2])]

    calMx = np.zeros((nMics, 2, harmHi - harmLo + 1),
                     dtype='complex')

    #inputStruct = InfImp['Iteration'][-1,-1]['Input'][0,0]
    inputStruct = InfImp
    specInfImp = inputStruct['meanSpectrum'].squeeze()
    specInfImp = specInfImp[harmLo:harmHi+1]
    calMx[:,0,:] = np.transpose(specInfImp[:,:nMics])
    #inputStruct = InfPipe['Iteration'][-1,-1]['Input'][0,0]
    inputStruct = InfPipe
    specInfPipe = inputStruct['meanSpectrum'].squeeze()
    specInfPipe = specInfPipe[harmLo:harmHi+1]
    calMx[:,1,:] = np.transpose(specInfPipe[:,:nMics])

    # calculate pressure at ref plane for infinite impedance
    # for the normal case where the microphone is some distance from the
    # infinite impedance and the transfter matrix is needed
    pressure = (specInfImp[:,0] / A_old[0,0,:])

    for frequencyCounter in range(calMx.shape[2]):
        # calculate the A_n1 terms
        A[:,0,frequencyCounter] = (calMx[:,0,frequencyCounter]/
                                   pressure[frequencyCounter])
        # initialize fitting coefficients
        coeff = []
        const = []
        newRowCoeff = np.zeros(nMics,dtype='complex')
        newRowCoeff[0] = 1
        coeff.append(newRowCoeff)
        newRowConst = (A[0,0,frequencyCounter] *
                       A[0,1,frequencyCounter] /
                       A_old[0,0,frequencyCounter])
        const.append(newRowConst)

        loadNo = 1
        for j in range(nMics):
            for k in range(j):
                newRowCoeff = np.zeros(nMics,dtype='complex')
                newRowCoeff[k] = (z0 * calMx[j,loadNo,frequencyCounter] /
                                  knownZ[loadNo][frequencyCounter])
                newRowCoeff[j] = (-z0 * calMx[k,loadNo,frequencyCounter] /
                                  knownZ[loadNo][frequencyCounter])
                coeff.append(newRowCoeff)
                newRowConst = (A[j,0,frequencyCounter] *
                               calMx[k,loadNo,frequencyCounter] -
                               A[k,0,frequencyCounter] *
                               calMx[j,loadNo,frequencyCounter])
                const.append(newRowConst)

        # obtain the A_n2 terms by least_squares
        A[:,1,frequencyCounter],_,_,_ = np.linalg.lstsq(np.array(coeff), 
                                                  np.array(const))

    return A

