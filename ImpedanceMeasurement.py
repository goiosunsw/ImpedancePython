"""
Functions to compute 1D input impedances of instruments 
from geometrical data of a linear resontor.

Uses global parameters contained in phys_params


"""

import numpy as np
import sys
import matplotlib.pyplot as pl
import Impedance as imp
import ImpedanceSynthesiser as imps
import scipy.signal as sig
from copy import deepcopy


def tfe_sig(y, x, *args, **kwargs):
    """estimate transfer function from x to y,
       see csd for calling convention"""
    fxy, sxy = sig.csd(y, x, *args, **kwargs)
    fxx, sxx = sig.csd(x, x, *args, **kwargs)
    return sxy / sxx, fxx


try:
    from matplotlib.mlab import psd, csd, cohere

    def tfe(y, x, *args, **kwargs):
        """estimate transfer function from x to y,
           see csd for calling convention"""
        sxy, fxy = csd(y, x, *args, **kwargs)
        sxx, fxx = psd(x, *args, **kwargs)
        return sxy / sxx, fxx


except ImportError:
    tfe = tfe_sig


def calculate_impedance_from_pressure(signals, sr, nwind=1024, ref_sensor_num=0):
    """
    calculates the uncorrected impedance

    (this is to be compared to a theoretical or known 
     impedance in order to calculate calibration factors)
    """
    slave_sensor_nums = signals.shape[1]
    slave_sensor_nums.discard(ref_sensor_num)

    l = self.duct.get_total_length()
    sensor_gains = []
    sensor_coh = []

    duct = self.load_model

    for sno in slave_sensor_nums:
        # calculate measured transfer functions
        tz, ff = tfe(x=signals[:,self.ref_sensor_num],
                     y=signals[:,sno], Fs=sr, NFFT=nwind)
        cz, ff = cohere(x=signals[:,self.ref_sensor_num],
                        y=signals[:,sno], Fs=sr, NFFT=nwind)
        
        # calculate theoretical transfer functions
        calmx_inv = []
        z0th = []

        for f in ff:
            cmx1 = duct.transfer_mx_at_freq(f,
                    from_pos=l-self.sensor_positions[self.ref_sensor_num],
                    to_pos=l)
            cmx2 = duct.transfer_mx_at_freq(f,
                    from_pos=l-self.sensor_positions[sno],
                    to_pos=l)
            calmx_inv.append(np.array([cmx1[0,:], cmx2[0,:]]))
            z0th.append(duct.get_input_impedance_at_freq(f, from_pos=l))

        calmx_inv = np.array(calmx_inv)
        

class Sensor(object):
    """
    Defines a sensor, characterising its parameters
    """

    def __init__(self, position=0.0,
                 pressure_sensitivity = 1.0,
                 flow_sensitivity = 0.0,
                 sensor_type = 'Microphone'):
        """
        initialise the sensor parameters:

        position: microphone postition along the duct
        pressure_sensitivity: v/Pa
        flow_sensitivity: v/(m^3/s)
        sensor_type: Microphone/ Pressure/ Hot Wire
        """
        self.position = position
        self.pressure_sens = pressure_sensitivity
        self.flow_sens = flow_senstitvity
        self.set_description()

    def set_description(self, description='', type='Microphone',
                         brand='Unknown', model='', serial_number=''):
        """
        Set string descriptors of the mcrophone

        (purely for information purposes)
        """
        self.description = ''
        self.type = sensor_type
        self.brand = 'Unknown'
        self.model = ''
        self.serial_number = ''

    def set_position(self, position):
        """
        Set the microphone position in m from duct port
        """
        self.position = position

    def get_position(self):
        """
        get the microphone position (m)
        """
        return self.position


class Calibration(object):
    """
    Defines a calibration object included known loads and
    corresponding measurements
    """

    def __init__(self, load_model, measurement=None,
                 sensor_set=None, nwind=1024, sr=48000):
        """
        define a new calibration, based on:
        * a sensor set (or None, use Calibration.set_sensor_positions)
        * a load_model (DuctImpedance object)
        * a set of measurements 
          (set of arrays size (n_samples*n_sensors)
        * analysis window (2*number of freq bins)
        * Sampling rate
        """
        self.load_model = load_model
        self.load_measurements = measurement
        if sensor_set is None:
            self.sensor_positions = []
        else:
            self.set_sensor_list(sensor_set)
            self.update_sensor_positions_from_sensor_list()
        self.ref_sensor_num = 0
        
        # set default signal parameters
        self.sr = sr
        self.nwind = nwind

    def add_sensor(self, sensor):
        """
        add a sensor
        """

        self.sensors.append(sensor)
        self.sort_sensors(key='position')

    def sort_sensors(self, key='position'):
        """
        sort the sensors based on one of its attributes
        (default: position)
        """
        positions = [getattr(xx, key) for xx in self.sensors]
        sensor_list = []
        self.sensor_positions = []
        for ii, pos in sorted(positions):
            sensor_list.append(self.sensors[ii])
            self.sensor_positions.append(pos)
        self.sensors = sensor_list

    def set_sensor_list(self, sensor_list):
        """
        set the list of sensors used in measurement / calibration

        (unit is meters)

        the list of sensor posisitons is set from the sensor list
        but is kept in an independent list. It can be set independently.
        The sensor list is only used for reference

        sensors will be re-sorted based on their positions
        """
        self.sensors = sensor_list
        self.sort_sensors()

    def set_sensor_positions(self, positions):
        """
        Sets the sensor positions without setting the sensor list.

        (sensor positions specified in meters from the measurement
        plane)
        """
        self.sensor_positions = (positions)

    def update_sensor_positions_from_sensor_list(self):
        self.sensor_positions = []
        for sens in self.sensors:
            self.sensor_positions.append(sens.position)

    def get_number_of_sensors(self):
        return len(self.sensor_positions)

    def add_calibration_signals(self, signals, sr=1.0):
        """
        add a calibration set, including (maybe) the input signal 
        and the measurements by each sensor.

        Number of channels in "sensors" must match number of sensors 
        in measurement object: 

            Measurement.get_number_of_sensors()
        """

        self.load_measurements.append(signals)
    
    def calculate_impedance(self, signals):
        """
        calculates the uncorrected impedance

        (this is to be compared to a theoretical or known 
         impedance in order to calculate calibration factors)
        """

        sr = self.sr
        nwind = self.nwind

        # define signals used to calculate tf from
        ref_sensor_pos = self.sensor_positions[self.ref_sensor_num]
        slave_sensor_nums = set(np.arange(self.get_number_of_sensors()))
        slave_sensor_nums.discard(self.ref_sensor_num)

        duct = self.load_model

        sensor_coh = []
        sensor_gains = []
        thtf= []
        mtf = []

        for sno in slave_sensor_nums:
            sensor_pos = self.sensor_positions[sno]
            
            # calculate measured transfer functions
            tz, ff = tfe(x=signals[:,self.ref_sensor_num],
                         y=signals[:,sno], Fs=sr, NFFT=nwind)
            cz, ff = cohere(x=signals[:,self.ref_sensor_num],
                            y=signals[:,sno], Fs=sr, NFFT=nwind)
            # get theoretical transfer functions
            tzth = duct.pressure_transfer_func(freq = ff,
                             from_pos=ref_sensor_pos,
                             to_pos=sensor_pos)
            
            # FIXME: phase is inverted: why?
            tzth = np.conj(tzth)
            gains = tzth / tz
            sensor_gains.append(gains)
            sensor_coh.append(cz)
            thtf.append(tzth)
            mtf.append(tz)

        return ff, sensor_gains, sensor_coh, mtf, thtf


class CalibrationSet(object):
    def __init__(self, calibrations=[]):
        self.calibrations = calibrations

    def add_calibration(self, cal):
        """
        add a calibration set
        """
        self.calibrations.append(cal)
