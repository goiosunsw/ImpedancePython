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
    saturation_pressure = 6.1078 * 10.**(7.5*temperature/(temperature+
                                                          constants['celsius_to_kelvin']))
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
    phys_param['speed_of_sound'] = 331.3 + np.sqrt(1 + temperature/ constants['celsius_to_kelvin']) \
                                         + 1.24 * humidity

class AcousticWorld(object):
    '''
    Set of acoustic constant to be used in a synthesised impedance
    '''
    def __init__(self,temp=25., humid= 0.5, press = 101325.):
        '''
        Set a new acoustic world object:
        
        temp: temperature in deg C
        humid: relative humidity 
        press: ambient pressure in Pa
        '''
        self.temperature = temp
        self.humidity = humid
        self.pressure = press
        
        
        #self.speed_of_sound = 343.2
        #self.medium_density =  1.2
        
        self._recalc()
        
    def _recalc(self):
        self.speed_of_sound = 331.3 * np.sqrt(1 + self.temperature/ 
                                              constants['celsius_to_kelvin']) \
                                    + 1.24 * self.humidity
        kelvin = self.temperature + constants['celsius_to_kelvin']
        
        vapour_pressure = water_vapour_pressure(self.temperature, 
                                                self.humidity)
        
        dry_pressure = self.pressure - vapour_pressure
        dry_density = dry_pressure / \
                      kelvin / \
                      constants['dry_air_gas_constant']
        
        vapour_density = vapour_pressure / \
                         kelvin /  \
                         constants['vapour_gas_constant']
        
        self.medium_density = vapour_density + dry_density
            
        # boundary layer constants give ratios are used to 
        # calculate rv and rt constants for effective speed calculations
        # rx = const*radius*f^1/2*
        # from Scavone : needs more work
        #self.viscous_boundary_layer_const = 632.8*(1-0.0029*self.temperature-300)
        #self.thermal_boundary_layer_const = 532.2*(1-0.0031*self.temperature-300)
        
        self.viscous_boundary_layer_const = np.sqrt(2*np.pi/self.speed_of_sound/4e-8)
        self.thermal_boundary_layer_const = np.sqrt(2*np.pi/self.speed_of_sound/5.6e-8)
        
       
class DuctSection(object):
    '''Any section connecting to another section in either end'''
    def __init__(self):
        '''Initialise a generic middle section portion
        
        Default is a section that does not change presure and flow
        (0-length cylinder)
        '''
        pass
        
    def _chain_reflection_coeff_at_freq(self, r_in, freq):
        '''Calculate reflection coefficient at beginning of section,
        when section is chained to a termination 
        with reflection coeff r_in'''
        
        tmx = self.travelling_mx_at_freq(freq)
        
        p_out = tmx[0,0]*1 + tmx[0,1]*r_in
        p_in  = tmx[1,0]*1 + tmx[1,1]*r_in  
        
        return p_in/p_out
            
    def _chain_impedance_at_freq(self, z_end, freq):
        '''Calculate reflection coefficient at beginning of section,
        when section is chained to a termination 
        with reflection coeff r_in'''
        
        tmx = self.transfer_mx_at_freq(freq)
        
        p_st = tmx[0,0]*z_end + tmx[0,1]*1
        u_st  = tmx[1,0]*z_end + tmx[1,1]*1  
        
        if p_st == np.inf:
            if u_st!=0.0:
                return np.inf
            else:
                raise ZeroDivisionError
        
        if u_st!=0.0:
            return p_st/u_st
        else:
            return np.inf
        
            
    def travelling_mx_at_freq(self, freq=0.0):
        ''' return the transfer matrix of the section 
        at a given frequency value:
           relates [P+,P-] at each end'''


        return np.array([[1,0],[0,1]])
    
    def transfer_mx_at_freq(self, freq=0.0):
        ''' return the transfer matrix of the section 
        at a given frequency value:
           relates [P,U] at each end'''

        return np.array([[1,0],[0,1]])
    
    def _recalc(self):
        pass
    
    def get_speed_of_sound(self):
        if self.parent:
            return self.parent.speed_of_sound
        else:
            return AcousticWorld().speed_of_sound

    def get_medium_density(self):
        if self.parent:
            return self.parent.medium_density
        else:
            return AcousticWorld().medium_density
            
    def get_boundary_layer_constants(self):
        if self.parent:
            world = self.parent.world
            
        else:
            world = AcousticWorld()
        return world.viscous_boundary_layer_const,\
               world.thermal_boundary_layer_const
    
    def set_parent(self, parent):
        self.parent=parent
        self._recalc()


class StraightDuct(DuctSection):
    def __init__(self, length = 0.5, radius = 0.1):
        self.radius = radius
        self.length = length
        self.parent = None
        
        self._recalc()
        self.gamma = 1.4
        
    def _recalc(self):
        self.cross_section = np.pi*self.get_input_radius()**2
        self.char_impedance = self.get_characteristic_impedance()
        rvc_per_rad, rtc_per_rad = self.get_boundary_layer_constants() 
        self.rv_const = rvc_per_rad * self.radius
        self.rt_const = rtc_per_rad * self.radius
        if self.parent:
            self.losses=self.parent.losses
        else:
            self.losses=True
    
    def get_propagation_coefficient(self, freq):
        if not self.losses:
            return 2*np.pi*freq/self.get_speed_of_sound()
        else:
            return self._propagation_coeff(freq)

    
    def _propagation_coeff(self, freq):
        c = self.get_speed_of_sound()
        rho = self.get_medium_density()
        
        omega   = 2  * np.pi * freq
        k       = omega/c
        
        rv      = self.rv_const * np.sqrt(freq)
        rt      = self.rt_const * np.sqrt(freq)

        P       = np.sqrt(2)/rv
        PQ      = (self.gamma-1)*np.sqrt(2)/rt;
        Zv      = omega*rho*(P*(1.+3.*P/2.)+1j*(1.+P));
        Yt      = omega/(rho*c**2)*\
                        (PQ*(1.-PQ/(2*(self.gamma-1.)))+1j*(1.+PQ));
        
        # propagation constant
        G       = np.sqrt(Zv*Yt) 
        # characteristic impedance
        # Zeta    = np.sqrt(Zv/Yt)/s  
        
        return -1j*G

    
    def get_input_radius(self):
        return self.radius
        
    def get_output_radius(self):
        return self.radius
        
    def get_length(self):
        return self.length
            
    def travelling_mx_at_freq(self, freq=0.0):
        
        phase = 2*np.pi*freq*self.length/self.get_speed_of_sound()
        
        return np.array([[np.exp(1j*phase),0],[0,np.exp(-1j*phase)]])

    def transfer_mx_at_freq(self, freq=0.0):
        
        
        prop_coeff = self.get_propagation_coefficient(freq)
        phase = prop_coeff*self.length
        
        return np.array([[np.cos(phase),
                            1j*self.char_impedance*np.sin(phase)],
                         [1j/self.char_impedance*np.sin(phase),
                            np.cos(phase)]])
                            
    def get_characteristic_impedance(self):
        return self.get_medium_density()*\
                          self.get_speed_of_sound()/\
                          self.cross_section

        
class TerminationImpedance(DuctSection):
    '''Base class for a termination impedance
    default is an open termination'''
    def __init__(self):
        self.__call__ = np.vectorize(self._get_impedance_at_freq)
        
    def __call__(self, f):
        return f
        
    def _get_reflection_coeff_at_freq(self, freq):
        return -1.
        
    def _get_impedance_at_freq(self, f):
        r = self._get_reflection_coeff_at_freq(f)
        if r!=1.0:
            return (1.+r)/(1.-r)
        else:
            return np.inf
        
    def plot_impedance(self, fig=None, fmin=0.0, fmax=4000.0, npoints=200):
        if not fig:
            fig,ax=pl.subplots(2, sharex=True)
        fvec = np.linspace(fmin,fmax,npoints)
        zvec = np.array([self._get_impedance_at_freq(f) for f in fvec])
        
        ax[0].plot(fvec,np.abs(zvec))
        ax[1].plot(fvec,np.angle(zvec))
        return fig,ax
        

class PerfectOpenEnd(TerminationImpedance):
    pass

class PerfectClosedEnd(TerminationImpedance):
    '''Ideal open end impedance
    Load impedance Zl(f) = 0
    Reflection function R(f) = 1 '''
    def _get_reflection_coeff_at_freq(self,freq):
        return 1.

class PortImpedance(object):
    '''
    Main functionality for an object with an input port
    '''
    
    def __init__(self):
        pass
    
    def get_input_impedance_at_freq(self,f):
        '''
        Retrieve the input impedance at a particular value of frequency
        '''
        return 0.0

class Duct(PortImpedance):
    '''
    1-D duct object containing linear elements
    '''
    def __init__(self, world=None, losses=True):
        if not world:
            world = AcousticWorld()

        self.set_acoustic_world(world)
        self.elements = []
        self.termination = PerfectOpenEnd()
        self.world = world
        self.losses = losses
        
    def set_acoustic_world(self,world):
        self.speed_of_sound = world.speed_of_sound
        self.medium_density = world.medium_density
        
    def append_element(self, element):
        assert isinstance(element, DuctSection)
        element.set_parent(self)
        self.elements.append(element)
        
    def insert_element(self, element, pos):
        assert isinstance(element, DuctSection)
        element.set_parent(self)
        self.elements.insert(element, pos)
        
    def set_termination(self, term):
        assert isinstance(term, TerminationImpedance)
        self.termination = term
        
    def get_input_reflection_function_at_freq(self, f):
        r = self.termination._get_reflection_coeff_at_freq(f)
        for el in reversed(self.elements):
            r = el._chain_reflection_coeff_at_freq(r, f)
        return r
    
    def get_input_impedance_at_freq(self, f):
        z = self.termination._get_impedance_at_freq(f)
        for el in reversed(self.elements):
            z = el._chain_impedance_at_freq(z, f)
        return z
        
    def get_coords(self):
        old_x=0
        x=[]
        y=[]
        for el in self.elements:
            x.append(old_x)
            y.append(el.get_input_radius())
            x.append(x[-1]+el.get_length())
            y.append(el.get_output_radius())
            old_x = x[-1]

        return x,y
