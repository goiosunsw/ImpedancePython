"""
Functions to compute 1D input impedances of instruments
from geometrical data of a linear resontor.

Uses global parameters contained in phys_params
"""

import numpy as np
import sys
import logging
import matplotlib.pyplot as pl
import matplotlib as mpl
from ._impedance import Impedance as imp
from copy import copy
from scipy.special import struve, j1

phys_params = {'speed_of_sound': 343.2,
               'medium_density': 1.2}

constants = {'dry_air_molar_mass': 0.02897,   # kg/mol
             'water_molar_mass': 0.018016,    # kg/mol
             'gas_constant': 8.314,           # J/K/mol
             'dry_air_gas_constant': 287.058,
             'vapour_gas_constant': 461.495,
             'celsius_to_kelvin': 273.15}


def water_vapour_pressure(temperature=25., humidity=0.5):
    saturation_pressure = 6.1078 * 10.**(7.5 * temperature /
                                         (temperature +
                                          constants['celsius_to_kelvin']))
    return saturation_pressure * humidity


def calc_params(temperature=25., humidity=0.5, ambient_pressure=101325.):
    """Recalculates physical parameters based on ambient values of:
    * temperature (in C)
    * humidity (in fraction i.e. 0.5 rather than 50%)
    * ambient_pressure (in Pa)"""

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
    phys_param['speed_of_sound'] = 331.3 + np.sqrt(1 + temperature/
                                                   constants['celsius_to_kelvin']) \
                                         + 1.24 * humidity


#def transfer_to_travelling_mx(transfer, char_impedance=1.0):
#    transfer[0, 1] /= char_impedance
#    transfer[1, 0] *= char_impedance
#
#    travelling = np.ones((2, 2), dtype='complex128')
#    travelling[0, 0] = np.sum(transfer)/2
#    travelling[0, 1] = -np.diff(np.sum(transfer, axis=0))[0]/2
#    travelling[1, 0] = -np.diff(np.sum(transfer, axis=1))[0]/2
#    travelling[1, 1] = (np.sum(np.diag(transfer)) -
#                        np.sum(np.diag(np.flipud(transfer))))/2
#    return travelling
#

def transfer_to_travelling_conversion_mx():
    """
    Returns the matrix C used to convert
    normalized travelling wave variables [p_out, p_in]
    to normalized acoustics variables [p, Zc u]

      [p, Zc U] = M [p_out, p_in]

    where:
        * p: acoustic pressure
        * u: acoustic flow
        * p_out: outgoing pressure wave
        * p_in : incoming pressure wave
    """
    return np.array([[1, 1], [1, -1]])


def transfer_to_travelling_mx(transfer, char_impedance=1.0):
    """
    Convert a transfer matrix M to a travelling wave matrix T

    Note:
        M: [p2, Zc u2] = M [p1, Zc u1]
        T: [p_out_2, p_in_2] = T [p_out_1, p_in_1]
    """
    # transfer_to_travelling conversion matrix
    ttm = transfer_to_travelling_conversion_mx()
    ttmi = np.linalg.inv(ttm)
    if transfer.ndim == 2:
        trav = np.dot(transfer, ttmi)
        trav = np.dot(ttm, trav)
    else:
        trav = np.zeros(transfer.shape, dtype='complex')
        for ii in range(transfer.shape[2]):
            trav[:,:,ii] = np.dot(transfer[:,:,ii], ttmi)
            trav[:,:,ii] = np.dot(ttm, trav[:,:,ii])
    return trav


class AcousticWorld(object):
    """
    Set of acoustic constants to be used in a synthesised impedance
    """
    def __init__(self, temp=25., humid= 0.5, press = 101325.):
        """
        Set a new acoustic world object:

        temp: temperature in deg C
        humid: relative humidity
        press: ambient pressure in Pa
        """
        self.temperature = temp
        self.humidity = humid
        self.pressure = press

        # self.speed_of_sound = 343.2
        # self.medium_density =  1.2

        self._recalc()

    def _recalc(self):
        """
        recalculate derived quantities when defining new
        temperature, pressure or humidity
        """
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
            kelvin / \
            constants['vapour_gas_constant']

        self.medium_density = vapour_density + dry_density

        # boundary layer constants give ratios are used to
        # calculate rv and rt constants for effective speed calculations
        # rx = const*radius*f^1/2*
        # from Scavone : needs more work
        # self.viscous_boundary_layer_const =\
        #     632.8*(1-0.0029*self.temperature-300)
        # self.thermal_boundary_layer_const =\
        #     532.2*(1-0.0031*self.temperature-300)

        self.viscous_boundary_layer_const =\
            np.sqrt(2*np.pi/self.speed_of_sound/4e-8)
        self.thermal_boundary_layer_const =\
            np.sqrt(2*np.pi/self.speed_of_sound/5.6e-8)


class DuctSection(object):
    """Any section connecting to another section in either end"""
    def __init__(self):
        """Initialise a generic middle section portion

        Default is a section that does not change presure and flow
        (0-length cylinder)
        """
        self.length = 0.0
        self.radius = 1.0
        self.char_impedance = 1.
        self.cross_section = 1.
        self.normalized_impedance = 1.0
        self.impedance_multiplier = 1.0
        self.gamma = 1.4
        self.parent = None

    def get_input_radius(self):
        """
        return the input radius of the duct
        (same as DuctSection.get_radius_at_position(0.0))
        """
        return self.radius

    def get_output_radius(self):
        """
        return the output radius of the duct
        (same as DuctSection.get_radius_at_position(
            DuctSection.get_length()))
        """
        return self.radius

    def _reset_impedance(self):
        """
        recalculate the characteristic impedance of the
        straight duct based on the geometrical dimentions:

            Zc = rho * c / S
            S = pi * r^2
        """
        self.cross_section = np.pi*self.get_input_radius()**2
        self.char_impedance = self.get_characteristic_impedance()
        if self.parent:
            self.normalized_impedance =\
                self.char_impedance/self.parent.get_characteristic_impedance()
        else:
            self.normalized_impedance = 1.0
        self.impedance_multiplier = self.char_impedance /\
            self.normalized_impedance

    def get_length(self):
        """
        returns the length of this duct section
        """
        return self.length

    def _chain_reflection_coeff_at_freq(self, r_in, freq):
        """
        Calculate reflection coefficient at beginning of section,
        when section is chained to a termination
        with reflection coeff r_in
        """

        tmx = self.travelling_mx_at_freq(freq)

        p_out = tmx[0, 0]*1 + tmx[0, 1]*r_in
        p_in = tmx[1, 0]*1 + tmx[1, 1]*r_in

        return p_in/p_out

    def _chain_impedance_at_freq(self, z_end, freq,
                                 from_pos=0.0, to_pos=None,
                                 reverse=False):
        """
        Calculate impedance at beginning of section,
        when section is chained to a termination
        with impedance z_end
        """

        tmx = self.normalized_two_point_transfer_mx_at_freq(freq,
                                                            from_pos=from_pos,
                                                            to_pos=to_pos,
                                                            reverse=reverse)

        # set dummy variable to zero if z0 is infinite
        # (this will prevent nan for infinite impedances)
        one = np.isfinite(z_end).astype('complex')
        z_end[np.logical_not(one)] = 1.
        p_st = tmx[0, 0, :]*z_end + tmx[0, 1, :]*one
        u_st = tmx[1, 0, :]*z_end + tmx[1, 1, :]*one
        z = p_st/u_st
        return z

#        if np.isinf(p_st):
#            if np.isinf(u_st):
#                return tmx[0, 0] / tmx[1, 0]
#            else:
#                return np.inf
#
#        if u_st != 0.0:
#            return p_st/u_st
#        else:
#            return np.inf
#
    def get_characteristic_impedance(self):
        """
        return the characteristic impedance of the duct section
        """
        return self.get_medium_density() *\
            self.get_speed_of_sound() /\
            self.cross_section

    def travelling_mx_at_freq(self, freq=0.0, position=0):
        """
        return the transfer matrix of the section
        at a given frequency value:
         relates [p_out,p_in] at each end
        """
        return np.array([[1, 0], [0, 1]])

    def normalized_two_point_transfer_mx_at_freq(self, freq=0.0,
                                                 from_pos=0.0,
                                                 to_pos=None,
                                                 reverse=False):
        """
        return the transfer matrix of the section
        at a given frequency value:
           relates [P, Zc U] at each end
        """

        vec1 = freq**0
        vec0 = freq*0
        return np.array([[vec1, vec0], [vec0, vec1]])

    def _recalc(self):
        """
        recalculates internal variables in this section
        """
        self._reset_impedance()

    def get_speed_of_sound(self):
        """
        returns the speed of sound associated to this duct
        """
        if self.parent:
            return self.parent.speed_of_sound
        else:
            return AcousticWorld().speed_of_sound

    def get_medium_density(self):
        """
        returns the density of the medium associated to
        this duct section
        """
        if self.parent:
            return self.parent.medium_density
        else:
            return AcousticWorld().medium_density

    def get_boundary_layer_constants(self):
        """
        returns the two boudary layer constants rv and rt

        rv, rt = get_boundary_layer_constants()

        rv: viscous constant
        rt: thermal constant
        """
        if self.parent:
            world = self.parent.world

        else:
            world = AcousticWorld()
        return world.viscous_boundary_layer_const,\
            world.thermal_boundary_layer_const

    def set_parent(self, parent):
        """
        set the parent duct container

        (should be a member of Duct class)
        """
        assert isinstance(parent, Duct)

        self.parent = parent
        self._recalc()


class ConicalDuct(DuctSection):
    """
    Conical section of a duct
    """
    def __init__(self, length=0.5, radius_in=0.01, radius_out=0.1, 
                 loss_multiplier=None):
        """
        create a conical section

        parameters:
            * length (m)
            * radius_in (m): input radius
            * radius_out (m): output radius
            * loss_multiplier: increases viscothermal losses by factor
        """
        super(ConicalDuct, self).__init__()
        self.radius_in = radius_in
        self.radius_out = radius_out
        self.radius = .5 * (self.radius_in+self.radius_out)
        self.length = length

        self._recalc()
        self.loss_multiplier = loss_multiplier

    def _recalc(self):
        """
        recalculate internal variables:
            * characteristic impedance
            * boundary layer coefficients
        """

        self._reset_impedance()
        # self.cross_section = np.pi*self.get_input_radius()**2
        rvc_per_rad, rtc_per_rad = self.get_boundary_layer_constants()
        self.rv_const = rvc_per_rad * self.radius
        self.rt_const = rtc_per_rad * self.radius
        if self.parent:
            self.losses = self.parent.losses
        else:
            self.losses = True

    def get_propagation_coefficient(self, freq):
        """
        returns the propagation coefficient at given frequency

        the propagation coefficient is the equivalent of "k"
        as in exp(j k l))

        it includes a real part, close to omega/c
        and a positive imaginary part, corresponding to distributed
        losses
        """
        if not self.losses:
            return 2*np.pi*freq/self.get_speed_of_sound()
        else:
            return self._propagation_coeff(freq)

    def get_output_radius(self):
        return self.radius_out

    def get_radius_at_position(self, position=0.0):
        """
        return the radius of the element at a given position (in m)
        """
        length = self.get_length()
        r_in = self.radius_in
        drad = self.radius_out - r_in
        return drad*position/length + r_in

    def _propagation_coeff(self, freq):
        """
        calcualtion of the propagation coefficient at given frequency
        """
        c = self.get_speed_of_sound()
        rho = self.get_medium_density()

        omega = 2 * np.pi * freq
        # k = omega/c

        rv = self.rv_const * np.sqrt(freq)
        rt = self.rt_const * np.sqrt(freq)

        P = np.sqrt(2)/rv
        PQ = (self.gamma-1)*np.sqrt(2)/rt
        Zv = omega*rho*(P*(1.+3.*P/2.)+1j*(1.+P))
        Yt = omega/(rho*c**2) *\
                (PQ*(1.-PQ/(2*(self.gamma-1.)))+1j*(1.+PQ))

        # propagation constant

        G = np.sqrt(Zv*Yt)
        if self.loss_multiplier is None:
            # characteristic impedance
            # Zeta    = np.sqrt(Zv/Yt)/s
            return ((G)/1j)
        else:
            return np.imag(G)-1j*np.real(G)*self.loss_multiplier

    def travelling_mx_at_freq(self, freq=0.0):
        """
        return the travelling wave matrix T of the complete
        duct section.

        [p_out, p_in]_end = T [p_out, p_in]_start

        same as StraightDuct.normalized_two_point_transfer_mx_at_freq(
                    freq=freq, start_pos=0,
                    end_pos=StraightDuct.get_length())
        """
        return self.two_point_travelling_mx_at_freq(freq=freq)

    def two_point_travelling_mx_at_freq(self, freq=0.0,
                                        from_pos=0.0, to_pos=None):
        """
        return the travelling wave matrix T of the
        duct section between from_pos to end_pos.

        [p_out, p_in]_from_pos = T [p_out, p_in]_to_pos
        """

        if from_pos == 0.0:
            from_rad = self.get_input_radius()
        else:
            from_rad = self.get_radius_at_position(from_pos)

        if to_pos is None:
            to_pos = self.get_length()
            to_rad = self.get_output_radius()
        else:
            to_rad = self.get_radius_at_position(to_pos)

        # distance = to_pos-from_pos
        # phase = 2*np.pi*freq*distance/self.get_speed_of_sound()

        distance = to_pos-from_pos

        prop_coeff = self.get_propagation_coefficient(freq)
        phase = prop_coeff*distance

        rad_rat = to_rad/from_rad

        return np.array([[np.exp(1j*phase), np.zeros_like(phase)],
                         [np.zeros_like(phase), np.exp(-1j*phase)]]) * rad_rat

    def transfer_to_travelling_mx(self, freq=None, pos=0.0):
        """
        Return the matrix X corresponding to:

        +- -+     +-  -+
        | p |     | p+ |
        |   | = X |    |
        | u |     | p- |
        +- -+     +-  -+
        """
        
        if pos==0.0:
            rad = self.get_input_radius()
        else:
            rad = self.get_radius_at_position(pos)

        cross_sect = np.pi*rad**2
        # cone "characteristic impedance" p+/u+
        prop_coeff = self.get_propagation_coefficient(freq)
        ones = np.ones_like(prop_coeff)
        if self.radius_in == self.radius_out:
            ych = ones * self.normalized_impedance
        else:
            dist_apex_from = self.get_distance_to_apex(pos)
            phase_apex_inv = 1/(1j*prop_coeff*dist_apex_from)
            ych = (1+phase_apex_inv)/self.normalized_impedance
        return np.array([[ones,ones],[ych, -np.conjugate(ych)]])

    def transfer_mx_at_freq(self, freq=0.0):
        """
        return the transfer matrix M of the
        duct section between the two edges of the section.

        [p, u]_end = M [p, u]_start
        """
        mx = self.normalized_two_point_transfer_mx_at_freq(freq=freq)
        mx[0, 1] *= self.impedance_multiplier
        mx[1, 0] /= self.impedance_multiplier
        return mx

    def normalized_transfer_mx_at_freq(self, freq=0.0):
        """
        return the normalized transfer matrix M of the
        duct section between the two edges pf the section.

        [p, Zc u]_end = M [p, Zc u]_start
        """
        return self.normalized_two_point_transfer_mx_at_freq(freq=freq)

    def get_distance_to_apex(self, pos):
        """
        get the distance to apex of cone frustum
        """
        length = self.get_length()
        r_in = self.get_input_radius()
        r_out = self.get_output_radius()
        apex = r_in/(r_out-r_in)*length
        return apex+pos

    def normalized_two_point_transfer_mx_at_freq(self, freq=0.0,
                                                 from_pos=0.0,
                                                 to_pos=None,
                                                 reverse=False):

        """
        return the normalized transfer matrix M of the
        duct section between from_pos to end_pos.

        [p, Zc u]_from_pos = M [p, Zc u]_to_pos
        """
        if to_pos is None:
            to_pos = self.get_length()

        distance = to_pos-from_pos
        if reverse:
            distance = -distance

        prop_coeff = self.get_propagation_coefficient(freq)
        phase = -prop_coeff*distance

        cmx_to = self.transfer_to_travelling_mx(freq=freq,pos=to_pos)
        cmx_from = self.transfer_to_travelling_mx(freq=freq,pos=from_pos)
        trav_mx = self.two_point_travelling_mx_at_freq(freq=freq,
                from_pos=from_pos, to_pos=to_pos)
        final = np.ones_like(trav_mx)
        for ii in range(trav_mx.shape[2]):
            tmp = np.matmul(trav_mx[:,:,ii], cmx_from[:,:,ii])
            final[:,:,ii] = np.matmul(np.linalg.inv(cmx_to[:,:,ii]), tmp)

        return final
    def normalized_two_point_transfer_mx_at_freq_one_go(self, freq=0.0,
                                                 from_pos=0.0,
                                                 to_pos=None,
                                                 reverse=False):

        """
        return the normalized transfer matrix M of the
        duct section between from_pos to end_pos.

        [p, Zc u]_from_pos = M [p, Zc u]_to_pos
        """
        if to_pos is None:
            to_pos = self.get_length()

        distance = to_pos-from_pos
        if reverse:
            distance = -distance

        prop_coeff = self.get_propagation_coefficient(freq)
        phase = -prop_coeff*distance
        
        if self.radius_in == self.radius_out:
            rat_apex_dist = 1
            rat_apex_len = 0
            phase_apex_inv = 0
            phase_to_inv=0
            phase_mix_inv = 0
        else:
            dist_apex_from = self.get_distance_to_apex(from_pos)
            dist_apex_to = self.get_distance_to_apex(to_pos)
            rat_apex_dist = dist_apex_to/dist_apex_from
            rat_apex_len = distance/dist_apex_from
            
            phase_apex_inv = 1/(prop_coeff*dist_apex_from)
            phase_to_inv = 1/(prop_coeff*dist_apex_to)
            phase_mix_inv = distance/(prop_coeff*dist_apex_from**2)

        return np.array([[rat_apex_dist*(np.cos(phase) - phase_to_inv*np.sin(phase)),
                          1j*self.normalized_impedance*np.sin(phase)/rat_apex_dist],
                         [1j/self.normalized_impedance*(
                             (rat_apex_dist - phase_apex_inv**2)*np.sin(phase) + 
                             (phase_mix_inv) *np.cos(phase)),
                          (np.sin(phase)*phase_apex_inv+np.cos(phase))/rat_apex_dist]])


class StraightDuct(DuctSection):
    """
    Straight section of a duct
    """
    def __init__(self, length=0.5, radius=0.1, loss_multiplier=None):
        """
        create a straight section

        parameters:
            * length (m)
            * radius (m)
            * loss_multiplier: increases viscothermal losses by factor
        """
        super(StraightDuct, self).__init__()
        self.radius = radius
        self.length = length

        self._recalc()
        self.loss_multiplier = loss_multiplier

    def _recalc(self):
        """
        recalculate internal variables:
            * characteristic impedance
            * boundary layer coefficients
        """

        self._reset_impedance()
        # self.cross_section = np.pi*self.get_input_radius()**2
        rvc_per_rad, rtc_per_rad = self.get_boundary_layer_constants()
        self.rv_const = rvc_per_rad * self.radius
        self.rt_const = rtc_per_rad * self.radius
        if self.parent:
            self.losses = self.parent.losses
        else:
            self.losses = True

    def get_propagation_coefficient(self, freq):
        """
        returns the propagation coefficient at given frequency

        the propagation coefficient is the equivalent of "k"
        as in exp(j k l))

        it includes a real part, close to omega/c
        and a positive imaginary part, corresponding to distributed
        losses
        """
        if not self.losses:
            return 2*np.pi*freq/self.get_speed_of_sound()
        else:
            return self._propagation_coeff(freq)

    def get_radius_at_position(self, position=0.0):
        """
        return the radius of the element at a given position (in m)
        """
        return self.get_input_radius()

    def _propagation_coeff(self, freq):
        """
        calcualtion of the propagation coefficient at given frequency
        """
        c = self.get_speed_of_sound()
        rho = self.get_medium_density()

        omega = 2 * np.pi * freq
        # k = omega/c

        rv = self.rv_const * np.sqrt(freq)
        rt = self.rt_const * np.sqrt(freq)

        P = np.sqrt(2)/rv
        PQ = (self.gamma-1)*np.sqrt(2)/rt
        Zv = omega*rho*(P*(1.+3.*P/2.)+1j*(1.+P))
        Yt = omega/(rho*c**2) *\
                (PQ*(1.-PQ/(2*(self.gamma-1.)))+1j*(1.+PQ))

        # propagation constant

        G = np.sqrt(Zv*Yt)
        if self.loss_multiplier is None:
            # characteristic impedance
            # Zeta    = np.sqrt(Zv/Yt)/s
            return ((G)/1j)
        else:
            return np.imag(G)-1j*np.real(G)*self.loss_multiplier

    def travelling_mx_at_freq(self, freq=0.0):
        """
        return the travelling wave matrix T of the complete
        duct section.

        [p_out, p_in]_end = T [p_out, p_in]_start

        same as StraightDuct.normalized_two_point_transfer_mx_at_freq(
                    freq=freq, start_pos=0,
                    end_pos=StraightDuct.get_length())
        """
        return self.two_point_travelling_mx_at_freq(freq=freq)

    def two_point_travelling_mx_at_freq(self, freq=0.0,
                                        from_pos=0.0, to_pos=None):
        """
        return the travelling wave matrix T of the
        duct section between from_pos to end_pos.

        [p_out, p_in]_from_pos = T [p_out, p_in]_to_pos
        """

        if to_pos is None:
            to_pos = self.get_length()

        # distance = to_pos-from_pos
        # phase = 2*np.pi*freq*distance/self.get_speed_of_sound()

        distance = to_pos-from_pos

        prop_coeff = self.get_propagation_coefficient(freq)
        phase = prop_coeff*distance

        return np.array([[np.exp(1j*phase), 0],
                         [0, np.exp(-1j*phase)]])

    def transfer_mx_at_freq(self, freq=0.0):
        """
        return the transfer matrix M of the
        duct section between the two edges of the section.

        [p, u]_end = M [p, u]_start
        """
        mx = self.normalized_two_point_transfer_mx_at_freq(freq=freq)
        mx[0, 1] *= self.impedance_multiplier
        mx[1, 0] /= self.impedance_multiplier
        return mx

    def normalized_transfer_mx_at_freq(self, freq=0.0):
        """
        return the normalized transfer matrix M of the
        duct section between the two edges pf the section.

        [p, Zc u]_end = M [p, Zc u]_start
        """
        return self.normalized_two_point_transfer_mx_at_freq(freq=freq)

    def normalized_two_point_transfer_mx_at_freq(self, freq=0.0,
                                                 from_pos=0.0,
                                                 to_pos=None,
                                                 reverse=False):

        """
        return the normalized transfer matrix M of the
        duct section between from_pos to end_pos.

        [p, Zc u]_from_pos = M [p, Zc u]_to_pos
        """
        if to_pos is None:
            to_pos = self.get_length()

        distance = to_pos-from_pos
        if reverse:
            distance = -distance

        prop_coeff = self.get_propagation_coefficient(freq)
        phase = -prop_coeff*distance

        return np.array([[np.cos(phase),
                          1j*self.normalized_impedance*np.sin(phase)],
                         [1j/self.normalized_impedance*np.sin(phase),
                          np.cos(phase)]])


class TerminationImpedance(DuctSection):
    """Base class for a termination impedance
    default is an open termination"""
    def __init__(self):
        self.__call__ = np.vectorize(self._get_impedance_at_freq)
        self.radius = 1.0
        self.char_impedance = 1.0
        super(TerminationImpedance,self).__init__()

    def __call__(self, freq):
        """
        return the value of the impedance at the given frequency
        """
        return f

    def _get_reflection_coeff_at_freq(self, freq):
        """
        returns the value of the reflection coefficient (complex)
        at a given frequency
        """
        return -np.ones_like(freq).astype('complex')
        # return r
        # if r.shape == 0:
        #     return r[()]
        # else:
        #     return r

    def _get_impedance_at_freq(self, freq):
        """
        return the value of the impedance (complex)
        at given frequency
        """
        r = self._get_reflection_coeff_at_freq(freq)
        return (1.+r)/(1.-r)

    def plot_impedance(self, fig=None, fmin=0.0, fmax=4000.0, npoints=200):
        """
        plot the impedance as a function of frequency

        returns figure, axis
        """
        if not fig:
            fig, ax = pl.subplots(2, sharex=True)
        fvec = np.linspace(fmin, fmax, npoints)
        zvec = np.array([self._get_impedance_at_freq(f) for f in fvec])

        ax[0].plot(fvec, np.abs(zvec))
        ax[1].plot(fvec, np.angle(zvec))
        return fig, ax

    def far_field_transpedance(self, f):
        """
        Returns the farfield pressure*distance for a unit volume flow
        at the radiation plane

        to get the pressure at a particular point at r meters from duct
        output:
            p = FlangedPiston.far_field_transpedance() * flow / distance
                * exp(1j*k*distance)
        """
        return 1j/2*f*self.get_medium_density()


class PerfectOpenEnd(TerminationImpedance):
    """
    a perfect open end, imposing
        * p=0
        * r=-1
        * z=0
    at any given frequency
    """
    pass

class FlangedPiston(TerminationImpedance):
    def __init__(self, radius=None):
        super(FlangedPiston,self).__init__()
        self._own_radius = radius
        if radius is None:
            self.radius=0.1
        self._reset_impedance()
        self.approx=False

    def set_parent(self,parent):
        if self._own_radius is None:
            self.radius = parent.elements[-1].get_output_radius()
        else:
            self.radius = self._own_radius
        super(FlangedPiston,self).set_parent(parent)

    def _get_impedance_at_freq(self, f):
        c = self.get_speed_of_sound()
        K = 2*np.pi*f/c
        ka = K * self.radius
        # not sure that Z0 should be the parent one...
        Z0 = self.normalized_impedance

        if self.approx:
            zfletch = (((ka)**2/2)**-1+1)**-1 + \
                       1j*((8*ka/3/np.pi)**-1 + (2/np.pi/ka)**-1)**-1
        else:
            zfletch = 1-j1(2*ka)/ka + 1j*struve(1,2*ka)/ka


        Z_flange = Z0*zfletch

        return Z_flange

    def far_field_transpedance(self, f):
        """
        Returns the farfield pressure*distance for a unit volume flow
        at the radiation plane

        to get the pressure at a particular point at r meters from duct
        output:
            p = FlangedPiston.far_field_transpedance() * flow / distance
                * exp(1j*k*distance)
        """
        return 1j/2*f*self.get_medium_density()

class InterpolatedImpedance(TerminationImpedance):
    """
    Represents a measurement to be fitted to a model
    """

    def __init__(self, radius=0.1):
        super(InterpolatedImpedance,self).__init__()
        self._f = np.array([0,20000])
        self._z = np.array([0,0])
        self.radius = radius
        self.interp_kwargs = {'kind': 'linear'}
        self._reset_impedance()


    def _get_impedance_at_freq(self, f):
        return np.interp(f, self._f, self._z)

    def set_points(self, f, z):
        self._f = f
        self._z = z

class PerfectClosedEnd(TerminationImpedance):
    """
    a perfect closed end, imposing
        * u=0
        * r=1
        * z=infinity
    at any given frequency
    """
    def _get_reflection_coeff_at_freq(self, freq):
        return np.ones_like(freq)

class PerfectAnechoicEnd(TerminationImpedance):
    """
    Ideal open end impedance
    Load impedance Zl(f) = 0
    Reflection function R(f) = 1
    """
    def _get_reflection_coeff_at_freq(self, freq):
        return np.zeros_like(freq)


class PortImpedance(object):
    """
    Main functionality for an object with an input port
    """

    def __init__(self):
        pass

    def get_input_impedance_at_freq(self, freq):
        """
        Retrieve the input impedance at a particular value of frequency
        """
        return 0.0

    def get_input_impedance(self, fmin=50., fmax=5000., fstep=2., fvec=None):
        """
        Get an impedance object.

        either supply a frequency vector in fvec
        or frequency ranges as fmin:fstep:fmax
        """

        if fvec is None:
            fvec = np.arange(fmin,fmax,fstep)

        z = np.array([self.get_input_impedance_at_freq(f)
                      for f in fvec])
        return imp.Impedance(freq=fvec, imped=z)

    def plot_impedance(self, ax=None, fmin=1.0,
                       fmax=4000.0, npoints=200,
                       scale_type='db',label=None):
        """
        plot the impedance as a function of frequency

        returns axis (list)
        """
        if ax is None:
            newfig = True
            fig, ax = pl.subplots(2, sharex=True)
        else:
            newfig = False

        if label is None:
            label = ''

        fvec = np.linspace(fmin, fmax, npoints)
        # zvec = np.array([self.get_input_impedance_at_freq(f) for f in fvec])
        zvec = self.get_input_impedance_at_freq(fvec)

        if scale_type=='db':
            y = 20*np.log10(np.abs(zvec))
            ylabel = 'dB re. kg$\cdot$m$^{-4}\cdot$s$^{-1}$'
        else:
            y = np.abs(zvec)
            ylabel = 'kg$\cdot$m$^{-4}\cdot$s$^{-1}$'

        ax[0].plot(fvec, y, label=label)
        if scale_type == 'log' and newfig:
            ax[0].set_yscale('log')
        ax[1].plot(fvec, np.angle(zvec), label=label)

        if newfig:
            ax[0].set_ylabel(ylabel)
            ax[1].set_ylabel('phase (rad)')
            ax[1].set_xlabel('Frequency (Hz)')
        return ax


class Duct(PortImpedance):
    """
    1-D duct object containing linear elements
    """
    def __init__(self, world=None, losses=True):
        if not world:
            world = AcousticWorld()

        self.set_acoustic_world(world)
        self.elements = []
        self.element_positions = []
        self.termination = PerfectOpenEnd()
        self.world = world
        self.losses = losses
        self.char_impedance = 1.0

    def set_acoustic_world(self, world):
        self.speed_of_sound = world.speed_of_sound
        self.medium_density = world.medium_density

    def get_characteristic_impedance(self):
        return self.char_impedance

    def append_element(self, element):
        assert isinstance(element, DuctSection)
        # set duct characteristic impedance if first element
        if len(self.elements) == 0:
            self.char_impedance =\
                element.get_characteristic_impedance()
            self.reset_element_char_impedances()
        element.set_parent(self)
        self.elements.append(element)
        self.update_element_pos()

    def reset_element_char_impedances(self):
        try:
            self.termination._reset_impedance()
        except AttributeError:
            pass
        for el in self.elements:
            el._reset_impedance()

    def copy(self):
        new_duct = copy(self)
        new_duct.elements = copy(new_duct.elements)
        new_duct.element_positions = copy(new_duct.element_positions)
        return new_duct

    def reverse(self, termination=None):
        """
        return the reversed duct
        (by default with open end)
        """
        new_duct = Duct()
        for ii,el,st,ed in self.iter_elements_in_interval(
            from_pos=self.get_total_length(), to_pos=0.0):
            new_duct.append_element(el)

        if termination is None:
            termination = PerfectOpendEnd()
        new_duct.set_termination(termination)
        return new_duct

    def new_with_attached_load(self, load):
        """
        returns a new duct object with the load attached
        to the end of the current duct
        """
        new_duct = self.copy()

        if isinstance(load, Duct):
            for el in load.elements:
                new_duct.append_element(el)
            new_duct.set_termination(load.termination)
        else:
            new_duct.set_termination(load)

        return(new_duct)

    def insert_element(self, element, pos):
        assert isinstance(element, DuctSection)
        element.set_parent(self)
        self.elements.insert(element, pos)
        self.update_element_pos()

    def update_element_pos(self):
        new_positions = []
        start_position = 0.0
        for el in self.elements:
            new_positions.append(start_position)
            start_position += el.get_length()

        self.element_positions = new_positions

    def set_termination(self, term):
        assert isinstance(term, TerminationImpedance)
        term.set_parent(self)
        self.termination = term

    def get_input_reflection_function_at_freq(self, f):
        r = self.termination._get_reflection_coeff_at_freq(f)
        for el in reversed(self.elements):
            r = el._chain_reflection_coeff_at_freq(r, f)
        return r

    def iter_elements_in_interval(self, from_pos=0.0,
                                  to_pos=None,
                                  reverse=False):
        """
        Return the list of elements between two positions
        along the duct, along with starting and ending
        position in the duct start and end positions default to
        0. and None if the complete length of the segment
        is in the interval
        """
        if to_pos is None:
            to_pos=self.get_total_length()

        edges = [from_pos, to_pos]

        from_nbr, from_el = self.get_element_at_position(min(edges))
        to_nbr, to_el = self.get_element_at_position(max(edges))

        if from_pos > to_pos:
            reverse = True
        else:
            reverse = False

        if reverse:
            for el_nbr in range(to_nbr,from_nbr-1,-1):
                el = self.elements[el_nbr]
                el_from_pos = max(edges) - self.element_positions[el_nbr]
                el_len = el.get_length()
                if el_from_pos > el_len:
                    el_from_pos = el_len
                el_to_pos = min(edges) - self.element_positions[el_nbr]
                if el_to_pos < 0.:
                    el_to_pos = 0.
                yield el_nbr, el, el_from_pos, el_to_pos
        else:
            for el_nbr in range(from_nbr,to_nbr+1):
                el = self.elements[el_nbr]
                el_from_pos = min(edges) - self.element_positions[el_nbr]
                if el_from_pos < 0:
                    el_from_pos = 0.
                el_len = el.get_length()
                el_to_pos = max(edges) - self.element_positions[el_nbr]
                if el_to_pos > el_len:
                    el_to_pos = el_len
                yield el_nbr, el, el_from_pos, el_to_pos


    def get_input_impedance_at_freq(self, f, from_pos=0.0):
        el_nbr, el = self.get_element_at_position(from_pos)
        z = self.termination._get_impedance_at_freq(f)
        # elements are chained in reverse from termination
        elements = reversed(self.elements[el_nbr+1:])
        for el in elements:
            # transfer matrices also need to be reversed!
            z = el._chain_impedance_at_freq(z, f,  reverse=True)
        el = self.elements[el_nbr]
        rel_pos = from_pos - self.element_positions[el_nbr]
        z = el._chain_impedance_at_freq(z, f, from_pos=rel_pos, reverse=True)


        return z*self.char_impedance

    def as_interpolated_impedance(self, f=None):
        imp = InterpolatedImpedance(radius=self.get_radius_at_position(0.0))
        z0 = self.char_impedance
        imp.set_points(f,self.get_input_impedance_at_freq(f)/z0)
        return imp



    def get_coords(self):
        old_x = 0
        x = []
        y = []
        for el in self.elements:
            x.append(old_x)
            y.append(el.get_input_radius())
            x.append(x[-1]+el.get_length())
            y.append(el.get_output_radius())
            old_x = x[-1]

        return x, y

    def get_element_at_position(self, position=0.0):
        """
        returns the duct element at a given position
        along this duct

        returns: element_number, element

        element_number can be used to obtain:
            * the element, from Duct.elements[element_number]
            * its starting position, Duct.element_positions[element_number]
        """
        pos_el = zip(self.element_positions, self.elements)
        for order, (pos, el) in enumerate(pos_el):
            if pos+el.get_length() >= position:
                return order, el
        # out of bounds:
        return np.nan, None

    def get_radius_at_position(self, position=0.0):
        """
        return the radius of the duct at a given position
        (in meters)
        """
        el_nbr, el = self.get_element_at_position(position=position)
        relative_pos = position - self.element_positions[el_nbr]
        el_rad = el.get_radius_at_position(relative_pos)
        return el_rad

    def get_total_length(self):
        """
        returns the total length of the duct (sum of elements)
        """
        return (self.element_positions[-1] + self.elements[-1].get_length())

    def normalized_transfer_mx_at_freq(self, freq=0.0,
                                       from_pos=0.0, to_pos=None,
                                       reverse=True):
        """ Returns the pressure/ flow transfer matrix between
        two positions of the duct

        transfer_mx_at_freq(from_pos=0.0, to_pos=length)

        default to_pos = Duct.get_length()

        arguments:
            from_pos: position of source from upstream
            to_pos: position of source from upstream
            (negative position means relative to downstream)
            reverse: whether to chain the segments in reverse
                    order

        returns:
            2x2 matrix M such that:
                [p, Zc u]_from = [m11,m12; m21, m22] [p, Zc u]_to
        """

        try:
            freq.__iter__
        except AttributeError:
            freq = np.array([freq])

        mx = np.tile(np.identity(2),(len(freq),1,1)).swapaxes(0,2)

        for el_nbr, el, el_st_pos, el_end_pos in \
            self.iter_elements_in_interval(from_pos=from_pos, to_pos=to_pos,
                                           reverse=reverse):
            trans_mx = el.normalized_two_point_transfer_mx_at_freq
            #trans_mx = tmx(freq, from_pos=el_st_pos,
            #               to_pos=el_end_pos)
            mx = np.matmul(mx.swapaxes(0,2),
                           trans_mx(from_pos=el_st_pos,
                                    to_pos=el_end_pos,
                                    freq=freq).swapaxes(0,2)).swapaxes(0,2)

        # FIXME: not good when from and to are in the same duct and there are
        # further ducts downstream
        return mx

    def transfer_mx_at_freq(self, freq=0.0, from_pos=0.0, to_pos=None,
                                       reverse=True):
        """ Returns the pressure/ flow transfer matrix between
        two positions of the duct

        transfer_mx_at_freq(from_pos=0.0, to_pos=length)

        default to_pos = Duct.get_length()

        arguments:
            from_pos: position of source from upstream
            to_pos: position of source from upstream
            (negative position means relative to downstream)

        returns:
            2x2 matrix M such that:
                [p, u]_from = [m11,m12; m21, m22] [p, u]_to
        """
        mx = self.normalized_transfer_mx_at_freq(freq=freq,
                                                 from_pos=from_pos,
                                                 to_pos=to_pos,
                                                 reverse=reverse)

        # remove normalization of the impedance
        mx[0, 1] *= self.char_impedance
        mx[1, 0] /= self.char_impedance
        return mx

    def travelling_mx_at_freq(self, freq=0.0,
                              from_pos=0.0, to_pos=None):
        """ Returns the travelling pressure transfer matrix between
        two positions of the duct

        transfer_mx_at_freq(from_pos=0.0, to_pos=length)
        arguments:
            from_pos: position of source from upstream
            to_pos: position of source from upstream
            (negative position means relative to downstream)

        returns:
            2x2 matrix T such that:
                [po, pi]_to = [t11,t12; t21, t22] [po, pi]_from
                (po is the outgoing pressure wave at each position
                 pi the corresponding incoming wave)
        """
        transf = self.normalized_transfer_mx_at_freq(freq=freq,
                                                     from_pos=from_pos,
                                                     to_pos=to_pos)
        return transfer_to_travelling_mx(transf,
                                         char_impedance=1.0)#self.char_impedance)

    # sys.stderr.write('{}\n'.format(to_pos))
        if to_pos is None:
            total_length = self.get_total_length()
            # sys.stderr.write('\nsetting position to {}\n'.format(total_length))
            end_pos = total_length
        else:
            end_pos = to_pos

        start_nb, start_element = self.get_element_at_position(from_pos)
        end_nb, end_element = self.get_element_at_position(end_pos)
        from_pos_rel = from_pos - self.element_positions[start_nb]
        to_pos_rel = end_pos - self.element_positions[end_nb]

        start_trav_mx = start_element.two_point_travelling_mx_at_freq
        if start_nb == end_nb:
            mx = start_trav_mx(from_pos=from_pos_rel,
                               to_pos=to_pos_rel,
                               freq=freq)
        else:
            mx = start_trav_mx(from_pos=from_pos_rel,
                               freq=freq)
            for el in self.elements[start_nb+1:end_nb]:
                mx = np.dot(mx, el.travelling_mx_at_freq(freq=freq))
                end_trav_mx = el.two_point_travelling_mx_at_freq
                mx = np.dot(mx, end_trav_mx(from_pos=0.0,
                                            to_pos=to_pos_rel,
                                            freq=freq))
        return mx

    def var_transfer_func(self, freq=np.array([1.0]),
                          from_pos=0.0, to_pos=None,
                          var='pressure'):
        """
        get the ratios of pressures at two positions in the duct
        """
        total_length = self.get_total_length()
        if to_pos is None:
            # sys.stderr.write('\nsetting position to {}\n'.format(total_length))
            end_pos = total_length
        else:
            end_pos = to_pos


        tmx = self.transfer_mx_at_freq(freq, from_pos=from_pos,
                                        to_pos=to_pos)
        z0 = (self.get_input_impedance_at_freq(freq,
                                               from_pos=from_pos))

        # set dummy variable to zero if z0 is infinite
        # (this will prevent nan for infinite impedances)
        one = np.isfinite(z0)
        z0[np.logical_not(one)] = 1.

        imx = np.array([[1*one,z0],[1/z0,1*one]])
        trx = np.matmul(tmx.transpose(2,0,1),
                        imx.transpose(2,0,1)).transpose(1,2,0)

        if var=='pressure':
            return trx[0,0,:]
        elif var=='flow':
            return trx[1,1,:]
        elif var=='transpedance':
            return trx[0,1,:]
        elif var=='transmittance':
            return trx[1,0,:]
        else:
            return trx

    def pressure_transfer_func(self, freq=1.0, from_pos=0.0,
                               to_pos=None, ref_pos=None, var='pressure'):
        """
        get the ratios of pressures at two positions in the duct
        """
        total_length = self.get_total_length()
        if to_pos is None:
            # sys.stderr.write('\nsetting position to {}\n'.format(total_length))
            end_pos = total_length
        else:
            end_pos = to_pos

        if ref_pos is None:
            ref_pos = total_length

        cmx1 = self.transfer_mx_at_freq(freq, from_pos=ref_pos,
                                        to_pos=from_pos)
        cmx2 = self.transfer_mx_at_freq(freq, from_pos=ref_pos,
                                        to_pos=to_pos)
        z0 = (self.get_input_impedance_at_freq(freq,
                                               from_pos=ref_pos))

        # set dummy variable to zero if z0 is infinite
        # (this will prevent nan for infinite impedances)
        one = np.isfinite(z0)
        z0[np.logical_not(one)] = 1.
        if var=='pressure':
            tfp = (cmx2[0,0]*z0 + cmx2[0,1]*one) / \
                  (cmx1[0,0]*z0 + cmx1[0,1]*one)
        return tfp

    def radiation_transpedance(vt,f=[1]):
        """
        Return ratio of:
            * radiated pressure divided by
            * input flow
        """
        #zl = vt.termination._get_impedance_at_freq(f)
        zi = self.get_input_impedance_at_freq(f)
        tm = self.transfer_mx_at_freq(f,reverse=False)
        transp = np.zeros(len(f),dtype='complex')
        for ii, ff in enumerate(f):
            #aout = (np.dot(np.linalg.inv(tm[:,:,ii]),np.array([zl[ii],1])))
            #transp[ii]=(aout[0])
            aout = np.dot(tm[:,:,ii],np.array([zi[ii],1]))
            #aout = np.dot(tm[:,:,ii],np.array([0,1]))
            transp[ii] = aout[1] * vt.termination.far_field_transpedance(ff)
        return transp

    def plot_geometry(self, ax=None, vert=False):
        """
        plot a transverse section of the duct

        return the axis
        """
        if ax is None:
            newfig = True
            pl.figure()
            ax = pl.gca()
        else:
            newfig = False
        x, y = self.get_coords()
        if vert:
            ln = ax.plot(y,x)
            ax.plot(-np.array(y),x, color=ln[0].get_color())
        else:
            ln = ax.plot(x, y)
            ax.plot(x, -np.array(y), color=ln[0].get_color())
        if newfig:
            ax.set_xlabel('distance from input (m)')
            ax.set_ylabel('radial distance (m)')
        return ax

    def plot_acoustic_distribution(self, ax=None, n_freq=500,
                                   fmin=1.0, fmax=4000.,
                                   n_len = 200,
                                   var='pressure', x_ref=0, phase=False):
        """
        plot the distribution of pressure or flow along the duct,
        relative to the variable at x_ref

        use n_freq frequency values and n_len points along the duct
        """

        xmax = self.get_total_length()
        pos_vec = np.linspace(0, xmax, n_len)
        fvec = np.linspace(fmin, fmax, n_freq)
        tpl = [self.var_transfer_func(from_pos=x_ref,
                                      to_pos=pos,
                                      freq=fvec,
                                      var=var)
               for pos in pos_vec]
        tp = np.array(tpl)

        if ax is None:
            fig,ax = pl.subplots(1)

        #cmap=pl.get_cmap('coolwarm')
        cmap=pl.get_cmap('viridis')

        if phase:
            pvar = 20*np.log10(np.abs(tp))
            prang = np.max(pvar)-np.min(pvar)
            imgr = (pvar - np.min(pvar))/prang
            # imgr = (np.array([pvar,pvar,pvar]) -
            #        np.min(pvar))/prang
            # imgr = imgr.transpose(1,2,0)

            ph = np.mod(np.angle(tp),np.pi*2)/np.pi/2
            imghsv = np.array([ph,np.ones(imgr.shape),imgr]).transpose(1,2,0)
            img = mpl.colors.hsv_to_rgb(imghsv)

            ax.imshow(img,
                      extent=[fmin,fmax,0.,xmax],
                      aspect='auto', origin='lower')
        else:
            ax.imshow(20*np.log10(np.abs(tp)),
                      extent=[fmin,fmax,0.,xmax],
                      aspect='auto', origin='lower',cmap=cmap)



    def plot_report(self, ax=None, fmin=50.0,
                    fmax=2000, npoints=200, scale_type='log'):
        """
        plot a figure with geometry and impedance

        return a list of axes handles
        """
        if ax is None:
            ax = []
            fig=pl.figure()
            # dimensions
            lmarg = 0.05
            gwidth = .2
            gheight = .2
            chspace = 0
            cwspace = 0.05
            zmarg = 0.05
            zwidth = 0.7
            zheight = .5-zmarg-chspace/2
            ax.append(pl.axes([lmarg, .5-gheight/2,
                               gwidth, gheight]))
            ax.append(pl.axes([lmarg+gwidth+cwspace,
                               .5+chspace/2, zwidth, zheight]))
            ax.append(pl.axes([lmarg+gwidth+cwspace, zmarg, zwidth,
                               .5-zmarg-chspace/2], sharex=ax[-1]))
        self.plot_geometry(ax[0])
        ax[0].set_xlabel('distance from input (m)')
        ax[0].set_ylabel('radial distance (m)')

        self.plot_impedance(ax[1:], fmin=fmin,
                            fmax=fmax, npoints=npoints,
                            scale_type=scale_type)
        ylabel = 'kg$\cdot$m$^{-4}\cdot$s$^{-1}$'
        ax[1].set_ylabel(ylabel)
        if scale_type == 'log':
            ax[1].set_yscale('log')
        ax[2].set_ylabel('phase (rad)')
        ax[2].set_xlabel('Frequency (Hz)')

        return ax


def profile_to_duct(lengths=None, rad=None, x=None, reverse=False,
                    termination='piston', loss_multiplier=None):
    """
    Generate a duct object based on lists of cylindrical segments
    with lengths l and radii rad
    """
    vt = Duct()

    if lengths is None:
        lengths = np.diff(x)

    if not reverse:
        lengths = np.flipud(lengths)
        rad = np.flipud(rad)

    for ll, rr in zip(lengths, rad):
        if ll > 0:
            vt.append_element(StraightDuct(length=ll,
                                           radius=rr,
                                           loss_multiplier=loss_multiplier))
    if termination == 'piston':
        vt.set_termination(FlangedPiston(radius=rr))
    elif termination == 'open':
        vt.set_termination(PerfectOpenEnd())
    elif termination == 'closed':
        vt.set_termination(PerfectClosedEnd())

    return vt

def vocal_tract_reader(filename, columns=None,
                       skiprows=0,
                       unit_multiplier=1.,
                       loss_multiplier=1.,
                       n_segments=100):
    """
    reads a vocat tract file with two columns:
        * By default first column has lengths and second column has radii in m
        * Valid column names are:
            + lengths
            + positions (segment limits: first segment starts at 0)
            + radii
            + area
        * Choose unit multiplier to convert units to m. for example, 1000 means
        the units of the file is mm, and .5 means the units are in m but the
        column represents diameter

    returns a pympedance.Synthesiser.Duct object
    """
    import pandas as pd

    if columns:
        vtpd = pd.read_csv(filename, header=None, skiprows=skiprows)
    else:
        vtpd = pd.read_csv(filename)
        columns = vtpd.columns

    #print( vtpd)
    for ic, col in enumerate(columns):
        if col[:3].lower() == 'len':
            l = np.array(vtpd.iloc[:,ic].tolist())*unit_multiplier
            x = np.concatenate(([0],np.cumsum(l)))
        elif col[:3].lower() == 'pos' or col[:3].lower() == 'dis':
            x = np.array(vtpd.iloc[:,ic].tolist())*unit_multiplier
            x0 = np.concatenate(([0], x))
            l = np.diff(x0)
        elif col[:3].lower() == 'rad':
            r = np.array(vtpd.iloc[:,ic].tolist())*unit_multiplier
        elif col[:3].lower() == 'dia':
            r = np.array(vtpd.iloc[:,ic].tolist())*unit_multiplier/2
        elif col[:3].lower() == 'are':
            r = (np.array(vtpd.iloc[:,ic].tolist())/np.pi)**.5*unit_multiplier
        else:
            sys.stderr.write('Column {} ({}) skipped\n'.format(ic, col))

    #print(l)
    # find 0 or negative lengths, warn and remove them
#     nnp = (l <= 0)
#     if sum(nnp) > 0:
#         sys.stderr.write('Segments skipped:\n')
#         for ii in nnp:
#             if ii:
#                 sys.stderr.write('  {}: l={}, r={}\n'.format(ii, l[ii], r[ii]))
#         r = r[np.logical_not(nnp)]
#         l = l[np.logical_not(nnp)]
    
    #print(np.array([r,l]).T)
    l_tot = max(x)
    xpos = np.linspace(0,l_tot,n_segments+1)
    l_seg = l_tot/(n_segments)
    radii = np.interp(xpos,x,r)
    
    vt = Duct()
    vt.append_element(StraightDuct(length=l_seg/2,radius=radii[0],loss_multiplier=loss_multiplier))
    for rr in radii[1:-1]:
        vt.append_element(StraightDuct(length=l_seg,radius=rr,loss_multiplier=loss_multiplier))
    vt.append_element(StraightDuct(length=l_seg/2,radius=radii[-1],loss_multiplier=loss_multiplier))
    vt.set_termination(FlangedPiston(radius=radii[-1]))
    #vt.set_termination(psy.PerfectOpenEnd())
    #vt = psy.profile_to_duct(lengths=l, rad=r, reverse=True,loss_multiplier=2.0)#, nsegments=None)
    
    return vt
