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
                 sensor_type = 'Microphone', id=''):
        """
        initialise the sensor parameters:

        position: microphone postition along the duct
        pressure_sensitivity: v/Pa
        flow_sensitivity: v/(m^3/s)
        sensor_type: Microphone/ Pressure/ Hot Wire
        """
        self.position = position
        self.pressure_sens = pressure_sensitivity
        self.flow_sens = flow_sensitivity
        self.set_description(id=id)

    def set_description(self, id='', 
                        description='', 
                        sensor_type='Microphone',
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
        self.id = id

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


class SensorList(object):
    def __init__(self, sensor_list=[]):
        self.sensors = []
        self.set_list(sensor_list)
        self.sensor_dict = {ii:ss for ii,ss in enumerate(self.sensors)}
        self.ref_sensor_num = 0

    def __getitem__(self, idx):
        """
        get a sensor from the list
        idx can be an order number or id (string)
        """
        try:
            return self.sensors[idx]
        except TypeError:
            return self.sensors[self.sensor_dict[idx]]
        
        raise KeyError

    def append(self, sensor):
        """
        add a sensor
        """
        id = sensor.id
        self.sensors.append(sensor)
        try:
            sensor.id = 'mic%d' % int(id)
        except ValueError:
            pass
        self.sensor_dict[id] = self.sensors.index(sensor)

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

    def set_list(self, sensor_list):
        """
        set the list of sensors used in measurement / calibration

        (unit is meters)

        the list of sensor posisitons is set from the sensor list
        but is kept in an independent list. It can be set independently.
        The sensor list is only used for reference

        sensors will be re-sorted based on their positions
        """
        self.sensors = sensor_list
        #self.sort_sensors()

    def set_reference_number(self, num=0):
        """
        set the sensor used as a reference, i.e.
        aginst which others will be compared
        """
        self.ref_sensor_num = num

    def get_reference_num(self):
        """
        get the index of sensor used as reference (master)
        i.e., the one against which other transfer functions
        will be calculated
        """
        return self.ref_sensor_num

    def get_slave_list(self):
        """
        get list of non-reference sensors 
        (e.g. to calculate a series of transfer 
        functions)
        """
        sensor_idx = set(np.arange(self.get_number_of_sensors()))
        sensor_idx.discard(self.ref_sensor_num)
        return list(sensor_idx)

    def get_number_of_sensors(self):
        """
        returns the number of sensors in list
        """

        return len(self.sensors)

    def get_posisitons(self, indexes=None):
        """
        get a list with the positions of the sensors in m
        """
        pos = []
        if indexes is None:
            indexes = np.arange(self.get_number_of_sensors())
        for idx in indexes:
            sens = self.sensors[idx]
            pos.append(sens.get_position())
        return pos

    def set_positions(self, position_list=[]):
        """
        set the positions of the sensors

        argument is a list of sensor positions in meters
        they will be attributed in sequence to the sensors
        registered in the list.
        Extra positions will create a new sensor with default 
        properties
        """
        
        # nsens = self.get_number_of_sensors()
        try:
            indexes = position_list.keys()
        except AttributeError:
            indexes = range(len(position_list))
        for ii in indexes:
            pos = position_list[ii]
            try:
                self.sensors[ii].position = pos
            except (IndexError, KeyError):
                sens = Sensor(id=ii, position=pos)
                self.append(sens)


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
        * a load_model (Duct object)
        * a set of measurements 
          (set of arrays size (n_samples*n_sensors)
        * analysis window (2*number of freq bins)
        * Sampling rate
        """
        self.load_model = load_model
        if sensor_set is None:
            self.sensor_list = SensorList()
        else:
            self.sensor_list = sensor_set
        # set default signal parameters
        self.sr = sr
        self.nwind = nwind

    def set_sensor_positions(self, positions):
        """
        Sets the sensor positions without setting the sensor list.

        (sensor positions specified in meters from the measurement
        plane)
        """
        self.sensor_list.set_positions(positions)

    def update_sensor_positions_from_sensor_list(self):
        self.sensor_positions = []
        for sens in self.sensors:
            self.sensor_positions.append(sens.position)

    def get_number_of_sensors(self):
        return len(self.sensor_positions)

    def get_reference_id(self):
        ref = self.sensor_list.get_reference_num()
        return ref

    def get_slave_sensors(self):
        return self.sensor_list.get_slave_list()

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
        ref_num = self.get_reference_id()
        ref_sensor_pos = self.sensor_list[ref_num].get_position()
        slave_sensor_nums = self.get_slave_sensors()

        duct = self.load_model

        sensor_coh = []
        sensor_gains = []
        thtf= []
        mtf = []

        for sno in slave_sensor_nums:
            sensor_pos = self.sensor_list[sno].get_position()
            
            # calculate measured transfer functions
            tz, ff = tfe(x=signals[:,ref_num],
                         y=signals[:,sno], Fs=sr, NFFT=nwind)
            cz, ff = cohere(x=signals[:,ref_num],
                            y=signals[:,sno], Fs=sr, NFFT=nwind)
            # get theoretical transfer functions
            tzth = duct.pressure_transfer_func(freq=ff,
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


class ImpedanceHead(object):
    """
    defines an impedance head with basic acoustic system
    and sensor configuration
    """
    def __init__(self, duct=None, sensor_set=None):
        #self.duct = duct
        if sensor_set is None:
            self.sensor_set = SensorList()
        else:
            self.sensor_set = sensor_set

        if duct is None:
            l = .1
            r = .05
            duct = imps.Duct()
            duct.append_element(imps.StraightDuct(length=l,radius=r))
            self.base_geometry = duct
        else:
            self.base_geometry = duct
        self.calibration_set = []

    def set_geometry(self, duct):
        """
        set the base geometry of the impedance head
        (usually a straight duct)
        """
        self.base_geometry = duct

    def check_sensor_consistency(self):
        """
        checks that sensor positions fit in the impedance head
        """

        consistent = True
        head_length = self.base_geometry.get_total_length()
        for sensor in self.sensor_set:
            if sensor.get_position() > head_length:
                consistent = False
        return consistent

    def generate_calibration(self, load):
        """
        generates a calibration object by attaching the
        load to the calibration head
        """
        new_load = self.base_geometry.attach_load(load)
        return new_load

    def set_sensor_positions(self, pos):
        """
        set or change sensor positions
        (see SensorList.set_positions)
        """

        self.sensor.set.set_positions(pos)


class CalibrationSet(object):
    def __init__(self, calibrations=[], 
                 impedance_head=None):
        self.calibrations = calibrations
        self.sensor_list = [] 
        self.sensor_positions = []
        self.sensor_gains = []
        self.ref_sensor_num = 0
        self.impedance_head = impedance_head

    def add_calibration(self, cal):
        """
        add a calibration set, setting the sensor configuration
        """
        cal.set_sensor_list(self.sensor_list)
        self.calibrations.append(cal)

    def add_load(self, load):
        """
        add a calibration load, attaching it to the default 
        impedance head

        returns the matching calibration object
        (without any calibration data)
        """
        if impedance_head is None:
            raise AttributeError("""Base Impedance head not defined
            Define with CalibrationSet.set_head()""")
        else:
            cal = self.impedance_head.generate_calibration(load)
            self.add_calibration(cal)

        return cal

    def add_load_with_measurements(self, load, signals):
        """
        adds a calibration load with the matching measurements
        """
        cal = self.add_load(load)
        cal.add_calibration_signals(signals)