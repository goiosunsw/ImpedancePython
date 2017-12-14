"""
Functions to compute 1D input impedances of instruments 
from geometrical data of a linear resontor.

Uses global parameters contained in phys_params


"""

import numpy as np
import matplotlib.pyplot as pl
import ImpedanceSynthesiser as imps
import scipy.signal as sig
import warnings
import xmltodict


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


def calculate_impedance_from_pressure(signals, sr, 
                                      nwind=1024, 
                                      ref_sensor_num=0):
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

def load_head(filename):
    """
    loads impedance head data from a xml file
    """
    # build a hierarchical dictionary
    with open(filename,'r') as fid:
        head_dict = xmltodict.parse(fid)['impedance_head']

    head = ImpedanceHead()

    sensor_data=head_dict['sensors']
    sensor_set=SensorList()
    for i,s in enumerate(sensor_data):
        
        sensor = Sensor(id=s['id'],
                        pressure_sensitivity=float(s['pressure_sensitivity']),
                        flow_sensitivity=float(s['flow_sensitivity']),
                        position=float(s['position']))
        sensor.set_description(serial_number=s['serial_number'],
                               brand=s['brand'],
                               model=s['model'],
                               description=s['serial_number'])

        calib_data = s['calibration']
        fvec = []
        gains = []
        conf = []
        for c in calib_data['data']:
            fvec.append(float(c['freq']))
            try:
                gains.append(complex(c['gain']))
            except ValueError:
                gains.append(complex(np.nan))
            conf.append(float(c['confidence']))

        sensor.set_gains(fvec=fvec, gains=gains, confidence=conf)
        sensor_set.append(sensor)

    duct_dict = head_dict['duct']
    world = imps.AcousticWorld(temp=float(duct_dict['temperature']),
                               humid=float(duct_dict['humidity']),
                               press=float(duct_dict['pressure']))
    duct = imps.Duct(world=world)
    try:
        el_sort = sorted(duct_dict['elements'], 
                         key=lambda x:float(x['position']))
    except TypeError:
        el_sort = [duct_dict['elements']]

    for el in el_sort:
        # el = duct_dict['elements']
        eel = el['straight_duct']
        element = imps.StraightDuct(length=float(eel['length']),
                                    radius=float(eel['radius']))
        duct.append_element(element)

    return ImpedanceHead(duct=duct, sensor_set=sensor_set)



class CalibrationData(object):
    """
    Per-frequency calibration data
    """

    difftol = 0.0001
    
    def __init__(self, frequencies=None,
                 gains=None, sr=None, confidence=None, 
                 fstep=None):

        self._frequencies = None
        if frequencies is not None:
            self.frequencies = frequencies
        else:
            self.sr = float(sr)
            self.fstep = float(fstep)
            frequencies = self.frequencies
        
        if gains is not None:
            assert len(frequencies) == len(gains)
            self.gains = gains
        else: 
            self.gains = np.ones(len(frequencies))

        if confidence is not None:
            assert len(frequencies) == len(confidence)
            self.confidence = confidence
        else:
            self.confidence = np.ones(len(frequencies))
        # for later:
        self.admittance = np.zeros(len(frequencies))
    
    @property
    def frequencies(self):
        if self._frequencies is None:
            maxfreq = self.sr/2
            npoints = int(np.round(maxfreq/self.fstep))+1
            return np.linspace(0, maxfreq, num=npoints)
        else:
            return self._frequencies

    @frequencies.setter
    def frequencies(self, val):
        fdiff = np.diff(val)
        meandiff = np.mean(fdiff)
        if np.all(np.abs(fdiff/meandiff-1) < self.difftol):
            self.fstep = meandiff
            self._frequencies = None
        else:
            self.fstep = None
            self._frequencies = val

        self.sr = np.max(val)*2


class Sensor(object):
    """
    Defines a sensor, characterising its parameters
    """

    def __init__(self, position=0.0,
                 pressure_sensitivity=1.0,
                 flow_sensitivity=0.0,
                 sensor_type='Microphone', id=''):
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
        self.gains = None
        self.gains_confidence = None
        self.gains_fvec = None

    def set_gains(self, gains=None, confidence=None, fvec=None):
        """
        set the sensor gains and confidences at the frequencies
        given by fvec
        """
        assert len(fvec) == len(gains)
        self.calibration = CalibrationData(frequencies=np.array(fvec),
                                           gains=np.array(gains),
                                           confidence=np.array(confidence))
        
    def get_gains(self):
        """
        returns the freq. vector, gains and confidence 
        for the sensor
        """
        fvec = self.calibration.frequencies
        gains = self.calibration.gains
        conf = self.calibration.confidence

        return fvec, gains, conf

    def set_description(self, id='',
                        description='',
                        sensor_type='Microphone',
                        brand='Unknown', model='', serial_number=''):
        """
        Set string descriptors of the microphone

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
    def __init__(self, sensor_list=None):
        self.sensors = []
        if sensor_list is None:
            self.sensor_list = []
        else:
            self.set_list(sensor_list)
        self.sensor_dict = {ii: ss for ii, ss in enumerate(self.sensors)}
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

    def __len__(self):
        return len(self.sensors)

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

    def get_positions(self, indexes=None):
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
                 sensor_set=None, nwind=None, sr=None):
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

    def set_sensor_list(self, sensor_list):
        """
        Sets the sensor list.
        """
        if isinstance(sensor_list, SensorList):
            self.sensor_list = sensor_list
        else:
            self.sensor_list = SensorList(sensor_list)

    def get_sampling_rate(self):
        return self.sr

    def set_sampling_rate(self, sr):
        if self.sr is not None:
            warnings.warn('Signals already present. Resetting the sampling rate will break the calibration.\nPrevious sample rate %f'% self.sr)
        self.sr = sr

    def set_window_length(self, nwind):
        self.nwind = nwind

    def get_window_length(self):
        return self.nwind

    def update_sensor_positions_from_sensor_list(self):
        self.sensor_positions = []
        for sens in self.sensor_list:
            self.sensor_positions.append(sens.position)

    def get_number_of_sensors(self):
        """
        return the number of sensors associated with
        the calibration
        """
        return len(self.sensor_list)

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
        cal_sr = self.get_sampling_rate()
        if cal_sr is None:
            self.set_sampling_rate(sr)
        else:
            if cal_sr != sr:
                raise RuntimeError('Sample rate mismatch')


        # self.load_measurements.append(signals)
        f, gains, coh, _, _ = self.calculate_impedance(signals)
        self.gains = gains
        self.coherence = coh
    
    def calculate_impedance(self, signals):
        """
        calculates the uncorrected impedance

        (this is to be compared to a theoretical or known 
         impedance in order to calculate calibration factors)

        input: signals (N x Nbr of sensors)

        returns:
            freq_vector, sensor_gains, gain_coherence, 
              measured_transfer_funct, theoretical_tf

            * sensor gains, coherence and tf are given at freq_vector
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

        for sno in range(self.get_number_of_sensors()):
            sensor_pos = self.sensor_list[sno].get_position()
            
            # calculate measured transfer functions
            tz, ff = tfe(x=signals[:, ref_num],
                         y=signals[:, sno], Fs=sr, NFFT=nwind)
            cz, ff = cohere(x=signals[:, ref_num],
                            y=signals[:, sno], Fs=sr, NFFT=nwind)
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
        self.sensor_gains = []

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
        new_load = self.base_geometry.new_with_attached_load(load)
        return new_load

    def set_sensor_positions(self, pos):
        """
        set or change sensor positions
        (see SensorList.set_positions)
        """

        self.sensor_set.set_positions(pos)

    def get_sensor_positions(self):
        """
        get sensor positions
        (see SensorList.get_positions)
        """

        return self.sensor_set.get_positions()
    
    def signals_to_impedance_2mic(self, signals, sr=None,
                                  mics=None):
        """
        calculate impedance from measurement signals
        """

        if mics is None:
            mics = [0,1]

        assert len(mics) == 2, 'Microphone list should consist of two numbers' 
        for sno in mics:
            cal_sr = self.sensor_set[sno].calibration.sr
            if sr is None:
                sr = cal_sr

            assert sr == cal_sr, 'Sample rates do not match' 

        fstep = self.sensor_set[sno].calibration.fstep
        nwind = int(np.round(float(sr)/fstep))

        ref_no = mics[0]
        ch_no = mics[1]

        t, ff = tfe(x=signals[:, ref_no],
                    y=signals[:, ch_no],
                    Fs=sr, NFFT=nwind)
        c, ff = cohere(x=signals[:, ref_no],
                       y=signals[:, ch_no],
                       Fs=sr, NFFT=nwind)
        
        calmx_inv = self.get_calibration_matrix(from_mic=ref_no,
                                                to_mic=ch_no)

        zunk = ((calmx_inv[1, 0, :] -
                 t*calmx_inv[0, 0, :]) /
                (-calmx_inv[1, 1, :] +
                 t*calmx_inv[0, 1, :]))

        conf = np.min([c,
                       self.sensor_set[ref_no].calibration.confidence,
                       self.sensor_set[ch_no].calibration.confidence],
                      axis=0)
        
        # FIXME: why is the impedance inverted??
        return 1/zunk, conf

    def get_calibration_matrix(self, from_mic=None, to_mic=0):
        """
        return the calibration matrix between two microphones.

        the calibration matrix is defined as:
                  +         +
                  | a1,  b1 |
            C^-1= |         |
                  | a2,  b2 |
                  +         +
        such that 

           +     +        +    +
           | p'1 |        | p0 |
           |     | = C^-1 |    |
           | p'2 |        | u0 |
           +     +        +    +

        and that p'x are sensor measurements at positions x
        """
        
        if from_mic is None:
            from_mic = self.sensor_set.get_reference_num()

        from_sensor = self.sensor_set[from_mic]
        to_sensor = self.sensor_set[to_mic]
        pos_from = from_sensor.position
        pos_to = to_sensor.position
        l = self.base_geometry.get_total_length()

        gains = to_sensor.calibration.gains/from_sensor.calibration.gains

        f = from_sensor.calibration.frequencies
        cmx1 = self.base_geometry.transfer_mx_at_freq(f,
                                                      from_pos=pos_from,
                                                      to_pos=l)
        f = to_sensor.calibration.frequencies
        cmx2 = self.base_geometry.transfer_mx_at_freq(f,
                                                      from_pos=pos_to,
                                                      to_pos=l)
        calmx_inv = np.array([cmx1[0,:]*gains, cmx2[0,:]])

        return calmx_inv

    def save(self,filename):
        """
        saves impedance head data to a xml file
        """
        # build a hierarchical dictionary
        sensor_data = []
        for i,s in enumerate(self.sensor_set):
            calib_data = []
            cal_obj = self.sensor_set[i].calibration
            fvec = cal_obj.frequencies
            for ii,f in enumerate(fvec):
                calib_data.append({'freq': f,
                                   'gain': cal_obj.gains[ii],
                                   'confidence': cal_obj.confidence[ii]})

            sensor_data.append({'id':s.id,
                                'brand':s.brand,
                                'model':s.model,
                                'serial_number':s.serial_number,
                                'pressure_sensitivity':s.pressure_sens,
                                'flow_sensitivity':s.flow_sens,
                                'position':s.position,
                                'serial_number':s.serial_number,
                                'calibration':{'sr':cal_obj.sr,
                                               'data':calib_data}})
            duct = self.base_geometry
            el_data = []
            for ii,el in enumerate(duct.elements):
                el_data.append({'position': duct.element_positions[ii],
                                'straight_duct': {'radius': el.radius,
                                                 'length': el.length}})

            duct_dict = {'temperature':duct.world.temperature,
                         'humidity': duct.world.humidity,
                         'pressure': duct.world.pressure,
                         'elements':el_data,
                         'termination':duct.termination.__str__}

            imp_dict = {'impedance_head':{'duct':duct_dict,
                                          'sensors':sensor_data}}

            with open(filename,'w') as fid:
                fid.write(xmltodict.unparse(imp_dict))

class CalibrationSet(object):
    def __init__(self, calibrations=None,
                 impedance_head=None,
                 nwind=None, sr=None):
        """
        initialises the container for a set of calibrations

        There are two ways of populating the calibration:
        1. provide a set of calibration objects. These do not
           distinguish between impedance head and load. Sensor
           positions and gemoetry before last sensor
           should match (not checked for now)
        2. provide an impedance head and add loads (maybe with
           measurements). These are appended to the impedance head)
        """
        if calibrations is None:
            self.calibrations = []
        else:
            self.calibrations = calibrations
        #self.sensor_list = []
        #self.sensor_positions = []
        #self.sensor_gains = []
        self.ref_sensor_num = 0
        self.impedance_head = impedance_head
        self.nwind = nwind
        self.sr = sr
        self.coherence_to_weight_power = 16

    def add_calibration(self, cal):
        """
        add a calibration set, setting the sensor configuration
        """
        cal.set_sensor_list(self.get_sensors())
        nwind = cal.get_window_length()
        sr = cal.get_sampling_rate()
        if self.nwind is None:
            self.nwind = nwind
        else:
            cal.set_window_length(nwind)

        if self.sr is None:
            self.sr = sr
        else:
            if self.sr != sr:
                raise RuntimeError('Sample rate of calibration signals does not match project rate' )

        cal.set_window_length(self.nwind)
        cal.set_sampling_rate(self.sr)
        self.calibrations.append(cal)

    def add_load(self, load):
        """
        add a calibration load, attaching it to the default
        impedance head

        returns the matching calibration object
        (without any calibration data)
        also appends the generated calibration object
        to the calibration load
        """
        if self.impedance_head is None:
            raise AttributeError("""Base Impedance head not defined
            Define with CalibrationSet.set_head()""")
        else:
            cal_duct = self.impedance_head.generate_calibration(load)
            cal = Calibration(load_model=cal_duct)
            self.add_calibration(cal)

        return cal

    def add_load_with_measurements(self, load, signals, sr=None):
        """
        adds a calibration load with the matching measurements
        """
        cal = self.add_load(load)
        if sr is None:
            sr = self.get_sampling_rate()
        cal.add_calibration_signals(signals,sr=sr)

    def set_sensors(self, sensor_list):
        """
        set the sensor list

        sensor list can be a SensorList object
        or a list of sensors
        """
        if isinstance(sensor_list, SensorList):
            self.impedance_head.sensor_set = sensor_list
        else:
            self.impedance_head.sensor_set = SensorList(sensor_list)

    def set_sensor_positions(self, pos):
        """
        set the sensor list with position information only
        """
        sensor_list = self.get_sensors()
        sensor_list.set_positions(pos)

    def mix_calibrations(self, mic_nbr):
        """
        combines all calibration data for a given sensor
        to generate sensor parameters and confindence

        returns sensor gains (per frequency)
           and  confidence (0-1 per frequency)
        """
        power = self.coherence_to_weight_power

        fvec = self.get_frequency_vector()

        gg = np.zeros(len(fvec), dtype='complex')
        allw = np.zeros(len(fvec))
        weight_vec = []
        for cal_nbr, cal in enumerate(self.calibrations):
            weights = (cal.coherence[mic_nbr]**power)
            weight_vec.append(weights)
            allw += weights

        for cal_nbr, cal in enumerate(self.calibrations):
            weights = weight_vec[cal_nbr]/allw
            gg += (cal.gains[mic_nbr]) * weights
        
        weight_comb = np.max(np.array(weight_vec), axis=0)
        return gg, weight_comb

    def calibrate_gains(self):
        """
        calibrates gains for all sensors, setting the gains
        in the sensor list
        """

        fvec = self.get_frequency_vector()

        for sens_nbr, sensor in enumerate(self.get_sensors()):
            gains, weights = self.mix_calibrations(sens_nbr)
            sensor.set_gains(gains=gains, confidence=weights,
                             fvec=fvec)

    def update_sr_from_calibrations(self):
        """
        updates the value of the smapling rate from member
        calibrations and checks that they are all equal
        """
        sr = None
        for cal in self.calibrations:
            if sr is None:
                sr = cal.get_sampling_rate()
            else:
                cal_sr = cal.get_sampling_rate()
                if cal_sr:
                    assert(cal.get_sampling_rate() == sr)
        self.sr = sr

    def get_sampling_rate(self):
        """
        get the project sampling rate and update it from calibrations
        if needed
        """
        if self.sr is None:
            self.update_sr_from_calibrations()
        return self.sr

    def get_frequency_vector(self):
        """
        return a vector of frequencies corresponding to calibration
        vectors
        """
        sr = self.get_sampling_rate()
        return np.linspace(0,sr/2,int(self.nwind/2)+1)

    def get_sensor_positions(self):
        """
        get sensor positions associated with the 
        calibration set (stored in ImpedanceHead)
        """

        return self.impedance_head.get_sensor_positions()

    def get_sensors(self):
        """
        return the sensor list as a SensorList object
        """
        return self.impedance_head.sensor_set



