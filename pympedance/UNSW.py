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
import warnings
import numpy as np
from scipy.io import matlab
import scipy.signal as sig
from ._impedance import Impedance
from copy import copy, deepcopy
import logging
from itertools import combinations
from collections import OrderedDict
from .Synthesiser import InterpolatedImpedance

from scipy.linalg import pinv, lstsq

import pdb

def obj_extract_mics(obj, initial_number_of_channels=3,
                     use_mics = [0,1]):
    objMod = deepcopy(obj)
    for k in objMod.dtype.names:
        sh = objMod[k].shape
        
        idx = [slice(None) for ii in sh]
        try: 
            micDim = sh.index(initial_number_of_channels)
            idx[micDim] = useMics
        except ValueError:
            pass
            
            
        objMod[k] = objMod[k][idx]
        #print(k, sh)
        #print(k,objMod[k].shape)
    return objMod

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

def mat_parameter_from_file(matfile):
    """
    try to extract parameters from mat file
    """
    mdata = matlab.loadmat(matfile)
    parameters = None

    try:
        parameters = mdata['Parameters'][0,0]
    except KeyError:
        parameters = mdata
    try:
        parameters['harmLo']
        parameters['numPoints']
    except KeyError:
        raise
        return None

#    partup = namedtuple('parameters', parameters.dtype.names)
#    pp = partup
#    for k in parameters.dtype.names:
#        par = parameters[k]
#        if np.prod(par.shape) <= 1 or par.dtype[0] == 'U':
#            pp[k] = np.asscalar(par)
#        else:
#            pp[k] = par

    return parameters


class MeasurementParameters(object):
    def __init__(self, from_mat=None, num_points=1024, 
                 harm_lo=0, harm_hi=-1, window=None,
                 method='fft', nhop=None,
                 inf_imp_obj=None,
                 inf_pipe_obj=None):
        """
        Initialise parameters

        If a matlab structure is given in from_mat,
        it will be copied to the object

        from_mat should be read with scipy.io.loadmat(squeeze_me=False)
        """
        self.can_recalculate = True

        self.calib_files = {'inf_imp':None,
                            'inf_pipe':None,
                            'inf_flange':None}

        
        if from_mat is not None:
            self.load_mat_params(from_mat)
        else:
            self.num_points = num_points
            self.harm_lo = harm_lo
            self.harm_hi = harm_hi

        if window is None:
            self.window = 'rect'
        else:
            self.window = window

        if nhop is None:
            self.hop = self.num_points
        else:
            self.hop = nhop

        self.spec_method = method



    def calc_calibration_matrices_mic_pairs(self,
                                            infinite_imp_file,
                                            infinite_pipe_file,
                                            infinite_flange=None):
        """
        calculates a series of calibration matrices
        (as calc_calibration_marix() but per microphone pairs)

        in addition calculates quality indicators
        """
        inf_imp_obj = ImpedanceMeasurement(filename=infinite_imp_file,
                                           parameters=self)
        inf_pipe_obj = ImpedanceMeasurement(filename=infinite_pipe_file,
                                           parameters=self)
        inf_flange_obj = ImpedanceMeasurement(filename=infinite_flange_file,
                                           parameters=self)

        param_pairs = []
        channel_pairs = []
        for mic_pair in itertools.combinations(range(nMics),2):
            inf_imp_mod = inf_imp_obj.use_mics(mic_pair)
            inf_pipe_mod = inf_pipe_obj.use_mics(mic_pair)
            inf_flange_mod = inf_flange_obj.use_mics(mic_pair)

            param_mod = self.use_mics(mic_pair)
            new_a = param_mod.calc_calibration_matrix(inf_imp_mod,
                                                      inf_pipe_mod)
            param_mod.a = new_a
            param_pairs.append(param_mod)
            channel_pairs.append(mic_pair)




    def calc_calibration_matrix(self,
                               infinite_imp_file=None,
                               infinite_pipe_file=None,
                               infinite_flange_file=None,
                               infinite_imp_obj=None,
                               infinite_pipe_obj=None,
                               infinite_flange_obj=None):
        """
        Calculate an impedance matrix from raw calibration signals
        . uses the parameters in this structure

        Returns a calibration matrix A:
            [n_channels X 2 X n_freq_bins]
        """

        file_names = (infinite_imp_file, infinite_pipe_file,
                      infinite_flange_file)

        objects = (infinite_imp_obj, infinite_pipe_obj,
                      infinite_flange_obj)
       
        
        load_order = ('inf_imp','inf_pipe','inf_flange')
        

        files_given = (infinite_imp_file is not None and 
                       infinite_pipe_file is not None)
        
        calib_measurements = OrderedDict()
        calib_spectra = OrderedDict()

        A_old = self.A
        harmLo = self.harm_lo
        harmHi = self.harm_hi

        for load_name, load_file, load_obj in zip(load_order,
                                                  file_names,
                                                  objects):

            calib_measurements[load_name] = load_obj
            if load_obj is None:
                if load_file is not None:
                    calib_measurements[load_name] = \
                    ImpedanceMeasurement(filename=load_file,
                                         parameters=self,
                                         is_calibration=True)

                spec_name = 'spec_'+load_name
                try:
                    spec = calib_measurements[load_name].mean_spectrum
                    calib_spectra[spec_name] = spec[harmLo:harmHi+1]
                except AttributeError:
                    calib_spectra[spec_name] = None


        return self.calc_calibration_matrix_from_spectra(**calib_spectra)

    def calc_calibration_matrix_from_spectra(self,
                                             spec_inf_imp=None,
                                             spec_inf_pipe=None,
                                             spec_inf_flange=None):
        """
        Calculate the calibration matrix given the spectra
        obtained for the calibration loads:
            * infinite impedance
            * infinite pipe
        """

        load_order = ('inf_imp','inf_pipe','inf_flange')

        given_spec = {
            'inf_imp': spec_inf_imp,
            'inf_pipe': spec_inf_pipe,
            'inf_flange': spec_inf_flange}
        
        A_old = self.A
        A = copy(A_old)
        z0 = self.z0
        harmLo = self.harm_lo
        harmHi = self.harm_hi
        nChannelFirst = self.n_channel_first
        micSpacing = self.mic_pos
        nMics = len(micSpacing)
        # shouldn't it be...
        # nChannelLast = nChannelFirst + nMics
        nChannelLast = nMics
        fVec = self.frequency_vector

        known_impedances={
            'inf_imp': np.inf*np.ones(A_old.shape[2]),
            'inf_pipe': z0 * np.ones(A_old.shape[2]),
            'inf_flange': self.theoretical_flange(fVec)}
        
        calMx = []
        knownZ = []

        for load in load_order:
            v = given_spec[load]
            if v is not None:
                knownZ.append(known_impedances[load])

                calMx.append(v[:,:nMics])

        calMx = np.array(calMx).transpose((2,0,1))
        print(calMx.shape)

        # calculate pressure at ref plane for infinite impedance
        # for the normal case where the microphone is some distance from the
        # infinite impedance and the transfter matrix is needed
        pressure = (spec_inf_imp[:,0] / A_old[0,0,:])

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
                                                      np.array(const),rcond=-1)

        return A

    def use_mics(self, channel_list, 
                 inf_imp_obj=None, 
                 inf_pipe_obj=None,
                 inf_flange_obj=None):
        """
        return a copy of the iteration with selected channels
        """
        
        default_calib = self.get_calibration_objects()

        if inf_imp_obj is None:
            inf_imp_obj = default_calib['inf_imp']

        if inf_pipe_obj is None:
            inf_pipe_obj = default_calib['inf_pipe']

        if inf_flange_obj is None:
            inf_flange_obj = default_calib['inf_flange']

        new_param = copy(self)
        initial_number_of_channels = len(self.mic_pos)
        for k,v in new_param.__dict__.items():
            try:
                sh = new_param.__dict__[k].shape
            except AttributeError:
                sh = []
            
            idx = [slice(None) for ii in sh]
            try: 
                micDim = sh.index(initial_number_of_channels)
                idx[micDim] = channel_list
                new_param.__dict__[k] = self.__dict__[k][idx]
            except ValueError:
                pass

        if inf_imp_obj is not None and inf_pipe_obj is not None:
            inf_imp_mod = inf_imp_obj.use_mics(channel_list)
            inf_pipe_mod = inf_pipe_obj.use_mics(channel_list)
            new_a = new_param.calc_calibration_matrix(infinite_imp_obj=inf_imp_mod,
                                                      infinite_pipe_obj=inf_pipe_mod)
            new_param.a = new_a

        return new_param
        
 

    
    def load_mat_params(self, param_file):
        """
        read parameters from matlab structure Parameters
        from a matlab .MAT file
        """
        parameters = mat_parameter_from_file(param_file)
        
        # only store non-redundant parameters
        try:
            self.radius = np.asscalar(parameters['radius'])
            self.density = np.asscalar(parameters['rho'])
            self.speed_of_sound = np.asscalar(parameters['speedOfSound'])
        except KeyError:
            self.can_recalculate = False

        self.harm_lo = np.asscalar(parameters['harmLo'])
        self.harm_hi = np.asscalar(parameters['harmHi'])
        self.num_points = np.asscalar(parameters['numPoints'])
        self.sr = np.asscalar(parameters['samplingFreq'])
        try:
            self.n_channel_first = np.asscalar(parameters['nChannelFirst'])
        except KeyError:
            logging.warn('First head channel not defined, setting to 1')
            self.n_channel_first = 1
        # self.num_cycles = np.asscalar(parameters['numCycles'])
        try:
            self.mic_pos = np.squeeze(parameters['micSpacing'])
        except KeyError:
            logging.warning('microphone positions not known')
            self.mic_pos = None

        assert self.freq_lo == np.asscalar(parameters['freqLo'])
        assert self.freq_hi == np.asscalar(parameters['freqHi'])
        assert self.freq_incr == np.asscalar(parameters['freqIncr'])
        try:
            self.A = parameters['A'].squeeze()
        except (KeyError, ValueError):
            self.can_recalculate = False
            #logging.warn('Calibration matrix not found!')
            self.A = None

        #if not self.can_recalculate:
        #    logging.warn('Will not be able to recalculate impedances')

        param_path = os.path.split(param_file)[0]
        cal_path = os.path.join(param_path,'../calib/')

        self.find_calib_files(try_paths=[cal_path,param_path,'.'])

    def find_calib_files(self,
                         calib_dict={'inf_imp':'InfImpCalib.mat',
                                     'inf_pipe':'InfPipeCalib.mat',
                                     'inf_flange':'InfFlangeCalib.mat'},
                         try_paths=None):

        if try_paths is None:
            try_paths = ['.']

        tried = []
        for k,v in calib_dict.items():
            file_c = v
            for path in try_paths:
                cal_file = os.path.join(path, file_c)
                #self.inf_imp_obj = ImpedanceMeasurement(filename=cal_file,
                #                                        parameters=new_par)
                if os.path.isfile(cal_file):
                    self.calib_files[k] = cal_file
                    logging.info('using infinite impedance file %s'%cal_file)

    def get_calibration_objects(self):
        calib_dict = dict()
        for k,v in self.calib_files.items():
            if v is not None:
                calib_dict[k] = ImpedanceMeasurement(v)
            else:
                calib_dict[k] = None
        return calib_dict

    def bin_number_to_freq(self, bin_no):
        return self.sr/self.num_points * bin_no

    @property 
    def cross_section_area(self):
        return np.pi * self.radius ** 2

    @property 
    def z0(self):
        return self.density * self.speed_of_sound / self.cross_section_area

    @property
    def freq_lo(self):
        return self.bin_number_to_freq(self.harm_lo)

    @property
    def freq_hi(self):
        return self.bin_number_to_freq(self.harm_hi)

    @property
    def freq_incr(self):
        return self.bin_number_to_freq(1.)

    @property
    def frequency_vector(self):
        return np.arange(self.harm_lo, self.harm_hi+1) * (self.sr /
                                                       self.num_points)

    def theoretical_flange(self, freq=None):
        if freq is None:
            freq = self.frequency_vector()

        omega = 2*np.pi*freq
        ka = (omega/self.speed_of_sound) * self.radius
        R = 10**(2.0*np.log10(ka) - 0.3)
        X = 10**(0.99*np.log10(ka) - 0.09)
        z_load = R + 1j*X
        #z_load *= self.z0
        return z_load



class ImpedanceIteration(object):
    """
    Implements a single iteration of a measurement,
    with a particular output waveform and collected data
    """
    def __init__(self, input_signals, output_signal=None, 
                 n_loops=None, parameters=None):
        """
        start iteration object with the input (measured) signals,
        which is the response of the system to an output_signal.

        The output signal is looped n_loops times 

        input_signal is N_samples X N_channels
        """

        self.input_signals = input_signals
        self._output_signal = output_signal
        self.n_samples = self.input_signals.shape[0]
        self.n_channels = self.input_signals.shape[1]
        if n_loops is None:
            n_loops = np.floor(self.n_samples / 
                                      len(self.output_signal))
        self.n_loops = int(n_loops)
        #self.loop_n_samples = loop_n_samples
        if type(parameters) is str:
            self.param = MeasurementParameters(parameters)
        else:
            self.param = parameters

        # Minimum number of cycles needed for calculation
        # of spectral error
        self.min_cycles_for_error = 8
            
        
        #self.output_signal = output_signal

    @property
    def mean_impedance(self):
        param = self.param

        if not param.can_recalculate:
            raise ValueError('cannot recalculate: missing parameters')

        mean_input_spect, spectral_error = self.calc_mean_spectra(
                                             nwind=param.num_points,
                                             window=param.window,
                                             method=param.spec_method,
                                             nhop=param.hop)
        
        analysis = self.analyse_input(mean_input_spect, spectral_error)
        imped = analysis['p']/analysis['u']
        imped = imped.squeeze()
        self.z = imped
        return imped 

    @property
    def output_signal(self):
        return self._output_signal

    @output_signal.setter
    def output_signal(self, signal):
        if signal is not None:
            self.has_output=True
            self._output_signal = signal

        if self.n_samples < len(self.output_signal):
            raise ValueError("input_signal.shape[0] must be smaller than length of output")
        
        new_loop_n = np.floor(self.n_samples/len(self.output_signal)) 
        if self.n_loops != new_loop_n:
            warnings.warn("new number of loops adjusted to %d\n"
                          % new_loop_n)
            self.n_loops = new_loop_n

    @property
    def loop_length(self):
        if self.output_signal is not None:
            return len(self.output_signal)
        elif self.n_loops:
            return int(self.n_samples/self.n_loops)
        else:
            raise ValueError("unkown number of loops")

    @property
    def mean_waveform(self):
        sum_waveform = np.zeros((self.loop_length,self.input_signals.shape[1]))
        for ii in range(self.n_loops):
            ist = ii*self.loop_length
            iend = ist+self.loop_length
            sum_waveform += self.input_signals[ist:iend]
        return sum_waveform / self.n_loops

    def get_mic_spectra_per_loop(self,
                        nwind=1024,
                        window=None,
                        method='fft',
                        nhop=None):
        """
        chops wavefile into excitation loops and calculates spectra

        input_waves (N_samples x N_channels)

        optional:
            discard_loops: remove N loops from beginning and end

        returns an array with (samples_per_loop/2+1) X N_loops X N_channels
        """

        input_waves = self.input_signals

        nsamp = nwind
        if window is not None:
            wind = sig.get_window(window, nsamp)
        else:
            wind = np.ones(nsamp)

        if nhop is None:
            nhop = nsamp

        all_spec = []
        all_coh = []

        for chno in range(0,input_waves.shape[1]):
            chspec = []
            if method == 'fft':
                for n in range(0,input_waves.shape[0]-nsamp+1,nhop):
                    w = input_waves[n:n+nsamp,chno]
                    #wspec = np.fft.fft(w*wind)
                    wspec = waveform_to_spectrum(w*wind)
                    chspec.append(wspec[:int(nsamp/2+1)])
                all_spec.append(chspec)
            elif method == 'tf':
                wspec = tfe(y=input_waves[:,chno],
                            x=excitation_signal,
                            nfft=nsamp, nhop=nhop,
                            window=wind)
                wcoh = sig.coherence(y=input_waves[:,chno],
                            x=excitation_signal,
                            nfft=nsamp, nhop=nhop,
                            window=wind)
                all_spec.append([wspec])
                all_coh.append(wcoh)

        return np.array(all_spec).transpose((2,1,0))

    def use_mics(self, channel_list):
        """
        return a copy of the iteration with selected channels
        """
        new_iter = copy(self)
        new_iter.input_signals = self.input_signals[:,channel_list]
        return new_iter

    def calc_mean_spectra(self, discard_loops=0,
                        nwind=None,
                        window=None,
                        method='fft',
                        nhop=None):
        """
        Returns mean spectra and spectral error
        """

        if nwind is None:
            nwind = self.param.num_points
        total_spectrum = self.get_mic_spectra_per_loop(nwind=nwind,
                                                      window=window,
                                                      method=method,
                                                      nhop=nhop)
        if discard_loops>0:
            total_spectrum = total_spectrum[:,discard_loops:-discard_loops,:]
        mean_spectrum = np.nanmean(total_spectrum, axis=1)
        spectral_error = np.nanstd(total_spectrum, axis=1, ddof=1)
        num_cycles = total_spectrum.shape[1]
        
        self.mean_spectrum = mean_spectrum
        self.spectral_error = spectral_error
        return mean_spectrum, spectral_error

    def calc_sigma_var_spectral_error(self, spectral_error):
        """
        calculate a weighting variable based on spectral error
        """
        # Fit a function of the form Af^n to the noise data,
        # and replace the noise data with the (smooth) function.
        fvec = self.param.frequency_vector
        fit_data = np.array([np.ones(fvec.shape), np.log(fvec)]).T
        coeff,_,_,_ = lstsq(fit_data,np.log(spectral_error))
        spectral_error = np.exp(np.dot(fit_data, coeff))
        sigma_variable = np.transpose(np.array([spectral_error]),
                                     axes=[2,0,1])
        return sigma_variable

    def calc_sigma_var_no_error(self):
        """
        get a weighting function based on sqrt(freq)
        when no sepctral error is present
        """
 
        fvec = self.param.frequency_vector
        # otherwise, choose a noise function of f^(-0.5)
        sigma_variable = (fvec**(-0.5) *
                         np.ones((1,spectral_error.shape[1])))
        sigma_variable = np.transpose(np.array([sigma_variable]), 
                                     axes=[2,0,1])
        return sigma_variable

    def analyse_input(self, mean_spectrum, spectral_error, 
                      loop_no=-1, meas_type='Averaged'):
        """
        Calculate pressure and flow at reference plane
        """

        rcond = -1
        
        param = self.param
        num_cycles = self.n_loops
        noise_calculated = num_cycles >= self.min_cycles_for_error 
        #  calculate A and b matrices
        A = param.A
        harm_lo = param.harm_lo
        harm_hi = param.harm_hi
        n_channel_first = param.n_channel_first
        mic_spacing = param.mic_pos
        n_mics = len(mic_spacing)
        # shouldn't it be...
        # nChannelLast = nChannelFirst + nMics
        n_channel_last = n_mics
        fvec = param.frequency_vector

        spectral_error = self.spectral_error[harm_lo:harm_hi+1,
                                             n_channel_first-1:n_channel_last]

        if noise_calculated:
            sigma_variable = self.calc_sigma_var_spectral_error(spectral_error)
        else:
            sigma_variable = self.calc_sigma_var_no_error()
        # pdb.set_trace()
        if meas_type == 'Averaged':
            mean_spec_red = mean_spectrum[harm_lo:harm_hi+1,
                                          n_channel_first-1:n_channel_last]
            b = np.transpose(np.array([mean_spec_red]),
                             axes=[2,0,1])
        elif meas_type == 'Individual':
            # reorder the totalSpectrum vector to match meanSpectrum
            if n_mics == 1:
                temp_spectrum =\
                        total_spectrum[harm_lo:harm_hi+1,
                                       count_loops:count_loops+1, 0];
            elif n_mics == 2:
                temp_spectrum =\
                    [total_spectrum[harm_lo:harm_hi+1,
                                    count_loops:count_loops+1, 0],\
                     total_spectrum[harm_lo:harm_hi+1,
                                    count_loops:count_loops+1, 1]]
            else:
                temp_spectrum =\
                    [total_spectrum[harm_lo: harm_hi+1,
                                    count_loops:countLoops+1, 0],\
                    total_spectrum[harm_lo: harm_hi+1,
                                   count_loops:count_loops+1, 1],\
                    total_spectrum[harm_lo: harm_hi+1, count_loops:count_loops+1, 2]]

            b = np.transpose(temp_spectrum, axes=[0,2,1])

        # initialise p, u, deltap and deltau
        p = np.zeros((A.shape[2],1), dtype='complex')
        delta_p = np.zeros((A.shape[2],1), dtype='complex')
        u = np.zeros_like(p)
        delta_u = np.zeros_like(delta_p)
        for freq_count in range(A.shape[2]): # length of frequency vector
            # Calculate the covariant matrix by putting the elements of
            # sigmaVariable along the diagonal
            covar = np.diag(sigma_variable[:,0,freq_count]**2)
            # if only two mics, calculate x using backslash operator
            # i.e. solve A*x = B for x
            if n_mics == 1:
                x,_,_,_ = lstsq(A[:,:,freq_count], b[:,:,freq_count])
                # matrix is not square so use pseudoinverse
                Ainv = pinv(A[:,:,freqCount])
            elif n_mics == 2:
                x,_,_,_ = lstsq(A[:,:,freq_count], b[:,:,freq_count])
                Ainv = np.linalg.inv(A[:,:,freq_count])
            else:
                # otherwise, use a weighted least-squares to determine x
                # weighted least squares solution to A*x = b with weighting covar
                x,_,_,_ = lscov(A[:,:,freq_count], b[:,0,freq_count], covar,
                                rcond=rcond)
                Ainv = pinv(A[:,:,freq_count])
            # Use the inverse (or pseudoinverse) to calculate dx
            dx = np.sqrt(np.diag(np.dot(np.dot(Ainv, covar),
                                 Ainv.T)))
            # Calculate the impedance and its error
            Z0 = param.z0
            p[freq_count] = x[0]
            u[freq_count] = x[1] / Z0
            delta_p[freq_count] = dx[0]
            delta_u[freq_count] = dx[1] / Z0

        return dict(p=p, u=u, delta_p=delta_p, delta_u=delta_u, z0=Z0)




class ImpedanceMeasurement(object):
    """
    Implements an Impedance that can read matlab files measured in UNSW
    and do a range of operations on them

    (obsolete: use read_UNSW_impedance())
    """

    def __init__(self, filename=None, paramfile=None,
                 freqLo=None, freqIncr=1.0, parameters=None,
                 is_calibration=False):
        self.iterations = []
        self.parameters = None
        if filename is not None:
            fileext = os.path.basename(filename)
            base = os.path.splitext(fileext)
            self.name = os.path.basename(base[0])
            format = self.detectFormat(filename)
            if format == 'v6':
                self.readImpedance6(filename, paramfile=paramfile)

            elif format == 'v7':
                skip_params = False
                self.readImpedance(filename,skip_params=skip_params)
            else:
                raise IOError('Unrecognised format for the impedance file')

        if parameters is not None:
            self.parameters = parameters


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

    def readImpedance(self, filename, skip_params=False):
        # Read the mathfile
        mm = matlab.loadmat(filename)
        #self.parameters = mm['Parameters'][0,0]
        
        if not skip_params:
            self.parameters = MeasurementParameters(filename)


        miter = mm['Iteration'].flatten()
        out_sig = mm['Iteration'][0,0]['Output'][0,0]['waveform']

        self.iterations = []
        for ii in range(len(miter)):
            in_sig = miter[ii]['Input'][0,0]['originalWaveform']
            this_it = ImpedanceIteration(input_signals=in_sig,
                                  output_signal=out_sig,
                                  parameters=self.parameters)
        
            self.iterations.append(this_it)

        assert len(self.iterations) > 0, 'no iteration measurements found!'
        self.z = miter[-1]['meanZ'].squeeze()

    def get_channels_spectra(self, channel_list):
        """
        returns the chopped spectra corresponding to 
        channel_list
        """

        last_iter = self.iterations[-1]
        nwind = self.parameters.num_points
        all_spec = last_iter.get_mic_spectra_per_loop(nwind=nwind)
        return all_spec[:,:,channel_list]

    def get_mic_pair_set(self):
        all_mics = range(len(self.parameters.mic_pos))
        all_pairs = []
        pair_set=[]
        for mic_pair in combinations(all_mics,2):
            all_pairs.append(mic_pair)
            new_obj = self.use_mics(mic_pair)
            new_obj.calculate_impedance()
            pair_set.append(new_obj)

        return all_pairs, pair_set

    @property
    def f(self):
        return self.parameters.frequency_vector

    @property
    def mean_spectrum(self):
        last_iter = self.iterations[-1]
        mean_spec, spec_err = last_iter.calc_mean_spectra(discard_loops=1)
        return mean_spec

    @property
    def mean_waveform(self):
        last_iter = self.iterations[-1]
        return last_iter.mean_waveform


    def calculate_impedance(self):
        """
        recalculates the impedance from raw signals
        
        new impedance is stored in ImpedanceMeasurement.z
        """

        iteration = self.iterations[-1]
        self.z = iteration.mean_impedance
        return iteration.z

    def get_pressure_flow(self, average=True):
        """
        returns the spectral pressure and flow at the reference plane
        """

        iteration = self.iterations[-1]
        
        param = self.parameters
        mean_input_spect, spectral_error = iteration.calc_mean_spectra(
                                             nwind=param.num_points,
                                             window=param.window,
                                             method=param.spec_method,
                                             nhop=param.hop)
        analysis = iteration.analyse_input(mean_input_spect, spectral_error)
        return analysis['p'].squeeze(), analysis['u'].squeeze()


    def use_mics(self, channel_list, 
                 inf_imp_obj=None,
                 inf_pipe_obj=None):
        """
        returns new object with selected channels
        """

        new_param = self.parameters.use_mics(channel_list)
        new_it = []
        for it in self.iterations:
            it.param = new_param
            new_it.append(it.use_mics(channel_list))
            
        new_meas = ImpedanceMeasurement(parameters=new_param)
        new_meas.iterations = new_it
        return new_meas



    def readImpedance6(self, filename, paramfile=None):
        """
        Loads an Impedance 6 matlab structure.
        Parameter file is supplied in keyword argument paramfile
        or alternatively, supply freqLo and freqIncr to build 
        freqency vector
        """
        data = matlab.loadmat(filename)
        if paramfile is None:
            paramfile = self.lookForParams(filename)

        if paramfile is not None:
            parameters = MeasurementParameters(paramfile)
        else:
            # guess parameters from impedance file
            try:
                last_iter = data['iteration'].flatten()[-1]
                #spec = last_iter['input'][0,0]
                output = last_iter['output'][0,0]['waveform']
                num_points = output.shape[0]
                parameters = MeasurementParameters(num_points=num_points)
            except KeyError:
                logging.warn('could not guess parameters')
                parameters = MeasurementParameters()
            
        self.parameters = parameters
        miter = data['iteration'].flatten()
        last_iter = miter[-1]
        out_sig = data['iteration'][0,0]['output'][0,0]['waveform']
        
        for ii in range(len(miter)):
            in_sig = miter[ii]['input'][0,0]['waveform']
            this_it = ImpedanceIteration(input_signals=in_sig,
                                  output_signal=out_sig,
                                  parameters=self.parameters)
        
            self.iterations.append(this_it)


        
        try:
            z = last_iter['Z']
        except ValueError:
            z = last_iter['analysis'][0,0]['Z']

        self.z = z

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
                logging.warn('Could not find a parameter file')
                return None

    def as_interpolated_impedance(self, radius=None):
        if radius is None:
            radius = self.parameters.radius
        imp = InterpolatedImpedance(radius=radius)
        z0 = self.parameters.z0
        imp.set_points(self.f,self.z/z0)
        return imp


class BroadBandExcitation(object):
    """
    a broadband signal that repeats with an integer number of samples

    can be shaped to a desired spectrum profile
    """
    def __init__(self, n_points=1024, n_cycles=1):
        """
        creates a broadband "noise" signal object
        """
        self.n_points = n_points
        self.n_cycles = n_cycles
        self.harm_lo = harm_lo
        self.harm_hi = harm_hi
        self.spectrum = np.ones(int(n_points/2))
    
    def generate_cycle(self):
        """
        generates a single cycle of the the sound
        """
        spec = self.spectrum * 2*np.pi*np.rand.random(self.spectrum.shape[0])
        x = spectrum_to_waveform(self.spectrum, sel.n_points)
        


    def generate_sound(self):
        """
        generates the sound to a numpy 1 x n_points x n_cycles vector
        """

        cycle = self.generate_cycle()
        return np.tile(cycle, self.n_cycles)

def lscov(a, b, w, rcond=None):
    """
    calculates the weighted least squared solution to
      a.x = b
    given the weights w

    (equivalent to matlab lscov for a diagonal covariance matrix)
    """
    #t = np.sqrt(np.diag(w))
    t = np.linalg.cholesky(w)
    ta=np.linalg.solve(t,a)
    tb=np.linalg.solve(t,b)
    return np.linalg.lstsq(ta,tb,rcond=rcond)

# TODO:
# The result is slightly different when applying the analysis to
# an already recorded file, especially at high-freq.

def waveform_to_spectrum(waveform, num_points=None):
    if num_points is None:
        num_points = waveform.shape[0]
    spectrum = np.fft.fft(waveform, n=num_points, axis=0)
    # only keep non-trivial points
    harmHi = int(np.floor(num_points/2))
    spectrum = spectrum[:harmHi+1]
    # normalise
    spectrum = spectrum * 2 / num_points
    # fix DC
    spectrum[0] = spectrum[0] / 2

    return spectrum

def spectrum_to_waveform(spectrum, num_points=None, harm_lo=0):
    """
    spectrumtowaveform Calculates the waveform for a given spectrum.
    The ifft is performed on the columns of spectrum.

    note this uses the Matlab function spectrum that will be removed in a
    future release (after 2014b) - so needs to be updated
    """

    if num_points is None:
        harm_hi = spectrum.shape[0]
        num_points = (harm_hi-1)*2
    else:
        harm_hi = int(np.floor(num_points / 2))+1
    
    try:
        spec = np.zeros((num_points,spectrum.shape[1]), dtype='complex')
    except IndexError:
        spec = np.zeros(num_points, dtype='complex')
    # throw out any freqeuncy components that will not contribute
    try:
        spec[harm_lo:harm_hi] = spectrum[:spec.shape[0]]
    except (IndexError, ValueError):
        max_spec = min(harm_hi-1,len(spectrum)+harm_lo)
        spec[harm_lo:max_spec] = spectrum

    # fix up the dc component
    spec[0] = spec[0] * 2
    # un-normalise the points
    spec = spec * num_points / 2
    # add the extra points
    new_points = spec[1:harm_hi-1]
    new_points = np.conj(new_points)
    new_points = np.flipud(new_points)
    spec[harm_hi:] = new_points
    # take the ifft
    waveform = np.fft.ifft(spec, axis=0)
    waveform = waveform.real
    return waveform

def build_mic_spectra(input_waves, 
                      excitation_signal=None,
                      discard_loops=0,
                      nwind=1024,
                      window=None,
                      method='fft',
                      nhop=None):
    """
    chops wavefile into excitation loops and calculates spectra

    input_waves (N_samples x N_channels)

    optional: 
        discard_loops: remove N loops from beginning and end

    returns Input dict with
    * 'totalWaveform' (chopped waveform)
    * 'meanSpectrum'
    * 'totalSpectrum' (individual spectra for each loop)
    * 'spectralError'
    """


    inputdict = dict()
    nsamp = nwind
    if window is not None:
        wind = sig.get_window(window, nsamp)
    else:
        wind = np.ones(nsamp)

    if nhop is None:
        nhop = nsamp

    all_spec = []
    all_coh = []

    for chno in range(0,input_waves.shape[1]):
        chspec = []
        if method == 'fft':
            for n in range(0,input_waves.shape[0]-nsamp+1,nhop):
                w = input_waves[n:n+nsamp,chno]
                #wspec = np.fft.fft(w*wind)
                wspec = waveform_to_spectrum(w*wind)
                chspec.append(wspec[:int(nsamp/2+1)])
            all_spec.append(chspec)
        elif method == 'tf':
            wspec = tfe(y=input_waves[:,chno],
                        x=excitation_signal,
                        nfft=nsamp, nhop=nhop,
                        window=wind)
            wcoh = sig.coherence(y=input_waves[:,chno],
                        x=excitation_signal,
                        nfft=nsamp, nhop=nhop,
                        window=wind)
            all_spec.append([wspec])
            all_coh.append(wcoh)


    inputdict['totalSpectrum'] = np.array(all_spec).transpose((2,1,0))
    if discard_loops>0:
        inputdict['totalSpectrum'] = inputdict['totalSpectrum'][:,discard_loops:-discard_loops,:]
    inputdict['meanSpectrum'] = np.nanmean(inputdict['totalSpectrum'], axis=1)
    inputdict['spectralError'] = np.nanstd(inputdict['totalSpectrum'], axis=1, ddof=1)
    inputdict['numCycles'] = inputdict['totalSpectrum'].shape[1]*np.ones(input_waves.shape[1])

    return inputdict

def analyseinput(Input, Parameters, 
                 measType='Averaged', 
                 countLoops=0):
    """
    analyseinput takes the spectra and determines pressure and flow (u) to
    save in Analysis structure
    """

    rcond = -1

    numCycles = Input['numCycles'].squeeze()
    noiseCalculated = numCycles[0] >= 8
    #  calculate A and b matrices
    A = Parameters['A'].squeeze();
    harmLo = np.asscalar(Parameters['harmLo'])
    harmHi = np.asscalar(Parameters['harmHi'])
    nChannelFirst = np.asscalar(Parameters['nChannelFirst'])
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
        coeff,_,_,_ = lstsq(fitData,np.log(spectralError))
        spectralError = np.exp(np.dot(fitData, coeff))
        sigmaVariable = np.transpose(np.array([spectralError]),
                                     axes=[2,0,1])
    else:
        # otherwise, choose a noise function of f^(-0.5)
        sigmaVariable = (fVec**(-0.5) *
                         np.ones((1,spectralError.shape[1])))
        sigmaVariable = np.transpose(np.array([sigmaVariable]), 
                                     axes=[2,0,1])
    # pdb.set_trace()
    if measType == 'Averaged':
        meanSpecRed = meanSpectrum[harmLo:harmHi+1,nChannelFirst-1:nChannelLast]
        b = np.transpose(np.array([meanSpecRed]),
                         axes=[2,0,1])
    elif measType == 'Individual':
        # reorder the totalSpectrum vector to match meanSpectrum
        if nMics == 1:
            tempSpectrum =\
                    totalSpectrum[harmLo:harmHi+1, countLoops:countLoops+1, 0];
        elif nMics == 2:
            tempSpectrum =\
                [totalSpectrum[harmLo:harmHi+1, countLoops:countLoops+1, 0],\
                 totalSpectrum[harmLo:harmHi+1, countLoops:countLoops+1, 1]]
        else:
            tempSpectrum =\
                [totalSpectrum[harmLo: harmHi+1, countLoops:countLoops+1, 0],\
                totalSpectrum[harmLo: harmHi+1, countLoops:countLoops+1, 1],\
                totalSpectrum[harmLo: harmHi+1, countLoops:countLoops+1, 2]]

        b = np.transpose(tempSpectrum, axes=[0,2,1])

    # initialise p, u, deltap and deltau
    p = np.zeros((A.shape[2],1), dtype='complex')
    deltap = np.zeros((A.shape[2],1), dtype='complex')
    u = np.zeros_like(p)
    deltau = np.zeros_like(deltap)
    for freqCount in range(A.shape[2]): # length of frequency vector
        # Calculate the covariant matrix by putting the elements of
        # sigmaVariable along the diagonal
        covar = np.diag(sigmaVariable[:,0,freqCount]**2)
        # if only two mics, calculate x using backslash operator
        # i.e. solve A*x = B for x
        if nMics == 1:
            x,_,_,_ = lstsq(A[:,:,freqCount], b[:,:,freqCount])
            # matrix is not square so use pseudoinverse
            Ainv = pinv(A[:,:,freqCount])
        elif nMics == 2:
            x,_,_,_ = lstsq(A[:,:,freqCount], b[:,:,freqCount])
            Ainv = np.linalg.inv(A[:,:,freqCount])
        else:
            # otherwise, use a weighted least-squares to determine x
            # weighted least squares solution to A*x = b with weighting covar
            x,_,_,_ = lscov(A[:,:,freqCount], b[:,0,freqCount], covar,
                            rcond=rcond)
            Ainv = pinv(A[:,:,freqCount])
        # Use the inverse (or pseudoinverse) to calculate dx
        dx = np.sqrt(np.diag(np.dot(np.dot(Ainv, covar),
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
                                                  np.array(const),rcond=-1)

    return A

def resample_calibration(Parameters, 
                         InfImp=None,
                         InfPipe=None,
                         resamp=1):
    """
    Resample (downsample only) a known calibration by decimating
    the calibration matrix

    resamp is the factor by which to resample the calibration 

    new measurements will be based on Parameters['numPoints']/resamp
    points
    """
    ParametersNew = deepcopy(Parameters)
    numPoints = np.asscalar(Parameters['numPoints'])
    harmLo = np.asscalar(Parameters['harmLo'])
    harmHi = np.asscalar(Parameters['harmHi'])

    old_to_new_bins = np.arange(0,numPoints+1,resamp)
    fvec_mask = np.logical_and(old_to_new_bins >= harmLo-1,
                               old_to_new_bins <= harmHi-1)

    old_to_new_fidx = old_to_new_bins[fvec_mask] - harmLo + 1
    #pdb.set_trace()
    ParametersNew['numPoints'] = int(numPoints / resamp) 
    ParametersNew['A'] = Parameters['A'][:,:,old_to_new_fidx]
    ParametersNew['A_old'] = Parameters['A_old'][:,:,old_to_new_fidx]
    ParametersNew['calibrationMatrix'] = Parameters['calibrationMatrix'][:,:,old_to_new_fidx]
    ParametersNew['pressure'] = Parameters['pressure'][old_to_new_fidx]
    ParametersNew['frequencyVector'] = Parameters['frequencyVector'][old_to_new_fidx]
    ParametersNew['k'] = Parameters['k'][old_to_new_fidx]
    ParametersNew['harmLo'] = np.flatnonzero(old_to_new_bins >=
                                             harmLo-1)[0]
    ParametersNew['harmHi'] = np.flatnonzero(old_to_new_bins <=
                                             harmHi-1)[-1]

    ParametersNew['freqLo'] = ParametersNew['frequencyVector'][0]
    ParametersNew['freqHi'] = ParametersNew['frequencyVector'][-1]


    ParametersNew['Output'][0,0]['spectrum'] = \
        Parameters['Output'][0,0]['spectrum'][::resamp]

    #ParametersNew['Output'][0,0]['waveform'] = spectrumToWaveform(
    #    Parameters['Output'][0,0]['spectrum'], numPoints)

    if InfPipe is not None and InfImp is not None:
        input_waves = np.array([InfPipe['Input'][0,0]['originalWaveform'],
                                InfPipe['Input'][0,0]['originalWaveform']])
        Input_mod = build_mic_spectra(input_waves,
                                      nwind=Parameters['numPoints'])
        A = recalc_calibration(InfPipe=Input_mod[0],
                               InfImp =Input_mod[1],
                               Parameters=Parameters)
        ParametersNew['A'] = A
    return ParametersNew, old_to_new_fidx 

    
