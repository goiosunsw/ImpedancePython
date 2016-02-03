"""
Functions to compute 1D input impedances of instruments 
from geometrical data of a linear resontor.

Uses global parameters contained in phys_params


"""

import numpy as np
import matplotlib.pyplot as pl



phys_params = {'speed_of_sound': 343.2,
               'medium_density': 1.2}
               
constants = {'dry_air_molar_mass': 0.02897,   # kg/mol 
             'water_molar_mass': 0.018016,    # kg/mol
             'gas_constant': 8.314,           # J/K/mol
             'dry_air_gas_constant': 287.058, 
             'vapour_gas_constant': 461.495,
             'celsius_to_kelvin': 273.15}

def water_vapour_pressure(temperature=25., humidity=0.5):
    saturation_pressure = 6.1078 * 10.**(7.5*temperature/(temperature+237.3))
    return saturation_pressure * humidity

def calc_params(temperature=25., humidity=0.5, ambient_pressure=101325.):
    '''Recalculates physical parameters based on ambient values of:
    * temperature (in C)
    * humidity (in fraction i.e. 0.5 rather than 50%)
    * ambient_pressure (in Pa)'''
    
    global phys_params
    global constants
    
    kelvin = temperature + constants['celsius_to_kelvin']
    
    vapour_pressure = water_vapour_pressure(temperature, humidity)
    
    dry_pressure = ambient_pressure - vapour_pressure
    dry_density = dry_pressure / kelvin /  constants['dry_air_gas_constant']
    
    vapour_density = vapour_pressure / kelvin /  constants['vapour_gas_constant']
    
    phys_params['medium_density'] = vapour_density + dry_density
    
    # simplistic, for now... (Wikipedia)
    #molar_ratio =  (1-humidity) + humidity * constants['water_molar_mass'] / constants['dry_air_molar_mass']
    #phys_param['speed_of_sound'] = (331.3 + 0.606 * temperature) / np.sqrt(molar_ratio)
    phys_param['speed_of_sound'] = 331.3 + np.sqrt(1 + temperature/ constants['celsius_to_kelvin']) 
                                         + 1.24 * humidity

class DuctSection:
    '''Base class for a 1D resonator section'''
    def __init__(self):
        pass
    
    
    def __str__(self):
        '''Prints the geometrical dimensions of this section'''
        pass
        
       
class MiddleSection(self):
    '''Any section connecting to another section in either end'''
    def __init__(self, **kwargs):
        if kwargs:
            self.__SetTravellingWaveTransferMatrix(**kwargs)
        else:
            self.__SetTravellingWaveTransferMatrix(tmOO=1.0, tmOI=0.0, tmIO=0.0, tmII=1.0)
        
    def ApplyTransferMatrix(self, inMattrix, freq):
        '''Calculates the transfer matrix to apply 
        to the downstream sections
        
        inMattrix is the downstream transfer mattrix
        freq is the frequency vector at which to apply the mattrix'''
        try:
            
            
    def TransferMattrixAtFreq(self, freq=0.0):
        try:
            functional_mx = [[self.tmOO, self.tmOI], 
                             [self.tmIO, self.tmII]]
            mx = [[el(freq) for el in row] for row in functional_mx]
            return np.mattrix(mx)
        except AttributeError:
            raise
    
    def __SetTravellingWaveTransferMatrix(self, **kwargs):
        '''Set the transfer matrix in terms of travelling wave coefficients
        
        MiddleSection.__SetTravellingWaveTransferMatrix(self, tmPP=1, tmUP=0, tmPU=0, tmUU=1)
            _    _ _            _ _    _
            | Pi | | tmPP  tmUP | | Po |
        M = |    | |            | |    |
            | Ui | | tmPU  tmUU | | Uo |
            -    - -            - -    -
        
        where Pi, Ui are the pressure and flow at input
        and Po, Uo at the output
        '''
        
        mattrix_coeffs = ('tmPP','tmUP','tmPU','tmUU',)
        
        for key, value in kwargs.items():
            if key in mattrix_coeffs:
            try:
                value(0.0)
                setattr(self, key, value)
            except TypeError:
                setattr(self, key, np.vectorize(lambda x: value))
        
class TerminationImpedance(DuctSection):
    '''Base class for a termination impedance'''
    def __init__(self):
        self.zl = np.vectorize(lambda x: 0.0)
        
    def __call__(self, freq):
        '''Return the complex value of the impedance at 
        frequency'''
        
        return self.zl(freq)
        
    def plot(self, fig=None, fmin=0.0, fmax=4000.0, npoints=200):
        if not fig:
            fig=pl.figure
        fvec = np.linspace(fmin,fmax,npoints)
        pl.plot(fvec,self(fvec))
        return fig
        

class PerfectOpenEnd(TerminationImpedance):
    '''Ideal open end impedance
    Load impedance Zl(f) = 0
    Reflection function R(f) = -1 '''
    def __init__(self):
        self.zl = np.vectorize(lambda x: 0.0)


class PerfectClosedEnd(TerminationImpedance):
    '''Ideal open end impedance
    Load impedance Zl(f) = 0
    Reflection function R(f) = 1 '''
    def __init__(self):
        self.zl = np.vectorize(lambda x: np.Inf)


class StraightSection(MiddleSection):
    '''A straight tube element, either conical or cylindrical'''
    def __init__(self, r_in, r_out, length):
        '''Initialise duct section, with parameters:
        * r_in, r_out: input and output radiuses
        * length: (of the duct section)'''
        self.r_in   = r_in
        self.r_out  = r_out
        self.length = length
        
        self.apex_distance = self.set_apex_distance()
        self.base_distance = self.set_base_distance()
        self.cone_half_angle = np.atan(self.r_in / self.apex_distance)
        
    def CalcTransferMatrix(self):
        global phys_params
        
        c = phys_params['speed_of_sound']
        
        apex_dist = self.apex_distance
        cone_length = self.length + apex_dist
        
        radius_ratio = self.r_in / self.r_out
        area_ratio = radius_ratio * radius_ratio
        
        apexial_wave_area = 2.*np.pi*self.apex_distance**2 * (1. - cos())
        
        # change here for wall losses
        jkL = lambda ff: 2j*np.pi*ff/c * self.length
        jkAD = lambda ff: 2j*np.pi*ff/c * apex_distance
        jkCL2 = lambda ff: 2j*np.pi*ff/c * cone_length * cone_length
        
        tmPP = np.vectorize(lambda ff: area_ratio*(apex_dist/cone_length * np.cos(jkL(ff)) - 
                                       apex_distance / (jkCL2(ff)) * np.sin(jkL(ff))))
        tmUP =  np.vectorize(lambda ff: (apex_distance/cone_length) * 1j * np.sin(jkL(ff)))
        tmPU = 
        tmUU = np.vectorize(lambda ff: (apex_distance/cone_length) * (
                                          np.cos(jkL(ff)) +
                                          1./ (jkAD(ff)) * np.sin(jkL(ff))))
        
        
class OneDResonator:
    '''A one dimensional resonator,
    incorporating several elements'''
    def __init__(self):
        self.middle_elements = []
        self.termination = None
        self.set_perfect_open_pipe()
        
    def set_perfect_open_pipe(self):
        '''Set the termination impedance to a static pressure
        (Zl = 0)'''
        self.termination = PerfectOpenEnd()
        
    def set_perfect_closed_pipe(self):
        '''Set the termination impedance to a rigid wall
        (Zl = Inf)'''
        self.termination = PerfectClosedEnd()
        
    def get_input_impedance(self):
        pass
        
    def draw(self):
        '''Draw resonator in a window'''
        fig = pl.figure()
        fig.canvas.set_window_title('Test')
        return fig
        
    def plot_impedance(self, fig=None, fmin=0.0, fmax=4000.0, npoints=200):
        '''Plot the input impedance of the complete resonator'''
        if not fig:
            fig=pl.figure
        fvec = np.linspace(fmin,fmax,npoints)
        pl.plot(fvec,self.get_input_impedance(fvec))
        return fig
        