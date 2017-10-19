"""
Functions to compute 1D input impedances of instruments 
from geometrical data of a linear resontor.

Uses global parameters contained in phys_params


"""

import numpy as np
import sys
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


def transfer_to_travelling_mx(transfer, char_impedance=1.0):
    transfer[0, 1] /= char_impedance
    transfer[1, 0] *= char_impedance

    travelling = np.ones((2, 2), dtype='complex128')
    travelling[0, 0] = np.sum(transfer)/2
    travelling[0, 1] = -np.diff(np.sum(transfer, axis=0))[0]/2
    travelling[1, 0] = -np.diff(np.sum(transfer, axis=1))[0]/2
    travelling[1, 1] = (np.sum(np.diag(transfer)) -
                        np.sum(np.diag(np.flipud(transfer))))/2
    return travelling

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
    '''Any section connecting to another section in either end'''
    def __init__(self):
        '''Initialise a generic middle section portion

        Default is a section that does not change presure and flow
        (0-length cylinder)
        '''
        self.length = 0.0
        self.char_impedance = 1.
        self.normalized_impedance = 1.0
        self.impedance_multiplier = 1.0
        self.parent = None

    def get_length(self):
        return self.length

    def _chain_reflection_coeff_at_freq(self, r_in, freq):
        '''Calculate reflection coefficient at beginning of section,
        when section is chained to a termination
        with reflection coeff r_in'''

        tmx = self.travelling_mx_at_freq(freq)

        p_out = tmx[0, 0]*1 + tmx[0, 1]*r_in
        p_in = tmx[1, 0]*1 + tmx[1, 1]*r_in

        return p_in/p_out

    def _chain_impedance_at_freq(self, z_end, freq):
        '''Calculate impedance at beginning of section,
        when section is chained to a termination
        with impedance z_end'''

        tmx = self.normalized_transfer_mx_at_freq(freq)

        # p_st = tmx[0, 0]*z_end + tmx[0, 1]*1
        # u_st = tmx[1, 0]*z_end + tmx[1, 1]*1
        p_st = tmx[0, 0]*z_end + tmx[0, 1]*1
        u_st = tmx[1, 0]*z_end + tmx[1, 1]*1

        if np.isinf(p_st):
            if np.isinf(u_st):
                return tmx[0, 0] / tmx[1, 0]
            else:
                return np.inf

        if u_st != 0.0:
            return p_st/u_st
        else:
            return np.inf

    def get_characteristic_impedance(self):
        return self.char_impedance

    def travelling_mx_at_freq(self, freq=0.0):
        ''' return the transfer matrix of the section
        at a given frequency value:
           relates [P+,P-] at each end'''


        return np.array([[1, 0], [0, 1]])

    def normalized_transfer_mx_at_freq(self, freq=0.0):
        ''' return the transfer matrix of the section
        at a given frequency value:
           relates [P,U] at each end'''

        return np.array([[1, 0], [0, 1]])

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
        self.parent = parent
        self._recalc()


class StraightDuct(DuctSection):
    def __init__(self, length=0.5, radius=0.1):
        super(StraightDuct, self).__init__()
        self.radius = radius
        self.length = length

        self._recalc()
        self.gamma = 1.4

    def _reset_impedance(self):
        self.cross_section = np.pi*self.get_input_radius()**2
        self.char_impedance = self.get_characteristic_impedance()
        if self.parent:
            self.normalized_impedance =\
                self.char_impedance/self.parent.get_characteristic_impedance()
        else:
            self.normalized_impedance = 1.0
        self.impedance_multiplier = self.char_impedance /\
            self.normalized_impedance

    def _recalc(self):
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
        if not self.losses:
            return 2*np.pi*freq/self.get_speed_of_sound()
        else:
            return self._propagation_coeff(freq)

    def get_radius_at_position(self, position=0.0):
        return self.get_input_radius()

    def _propagation_coeff(self, freq):
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
        # characteristic impedance
        # Zeta    = np.sqrt(Zv/Yt)/s

        return -1j*G

    def get_input_radius(self):
        return self.radius

    def get_output_radius(self):
        return self.radius

    def travelling_mx_at_freq(self, freq=0.0):
        return self.two_point_travelling_mx_at_freq(freq=freq)

    def two_point_travelling_mx_at_freq(self, freq=0.0,
                                        from_pos=0.0, to_pos=None):

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
        mx = self.normalized_two_point_transfer_mx_at_freq(freq=freq)
        mx[0, 1] *= self.impedance_multiplier
        mx[1, 0] /= self.impedance_multiplier
        return mx

    def normalized_transfer_mx_at_freq(self, freq=0.0):
        return self.normalized_two_point_transfer_mx_at_freq(freq=freq)

    def normalized_two_point_transfer_mx_at_freq(self, freq=0.0,
                                                 from_pos=0.0, 
                                                 to_pos=None):

        if to_pos is None:
            to_pos = self.get_length()

        distance = to_pos-from_pos

        prop_coeff = self.get_propagation_coefficient(freq)
        phase = prop_coeff*distance

        return np.array([[np.cos(phase),
                          1j*self.normalized_impedance*np.sin(phase)],
                         [1j/self.normalized_impedance*np.sin(phase),
                          np.cos(phase)]])

    def get_characteristic_impedance(self):
        return self.get_medium_density() *\
            self.get_speed_of_sound() /\
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
        if r != 1.0:
            return (1.+r)/(1.-r)
        else:
            return np.inf

    def plot_impedance(self, fig=None, fmin=0.0, fmax=4000.0, npoints=200):
        if not fig:
            fig, ax = pl.subplots(2, sharex=True)
        fvec = np.linspace(fmin, fmax, npoints)
        zvec = np.array([self._get_impedance_at_freq(f) for f in fvec])

        ax[0].plot(fvec, np.abs(zvec))
        ax[1].plot(fvec, np.angle(zvec))
        return fig, ax


class PerfectOpenEnd(TerminationImpedance):
    pass


class PerfectClosedEnd(TerminationImpedance):
    '''Ideal open end impedance
    Load impedance Zl(f) = 0
    Reflection function R(f) = 1 '''
    def _get_reflection_coeff_at_freq(self, freq):
        return 1.


class PortImpedance(object):
    '''
    Main functionality for an object with an input port
    '''

    def __init__(self):
        pass

    def get_input_impedance_at_freq(self, f):
        '''
        Retrieve the input impedance at a particular value of frequency
        '''
        return 0.0

    def plot_impedance(self, ax=None, fmin=0.0, 
                       fmax=4000.0, npoints=200,
                       scale_type='db'):
        if ax is None:
            newfig = True
            fig, ax = pl.subplots(2, sharex=True)
        else:
            newfig = False

        fvec = np.linspace(fmin, fmax, npoints)
        zvec = np.array([self.get_input_impedance_at_freq(f) for f in fvec])
        
        if scale_type=='db':
            y = 20*np.log10(np.abs(zvec))
            ylabel = 'dB re. kg$\cdot$m$^{-4}\cdot$s$^{-1}$'
        else:
            y = np.abs(zvec)
            ylabel = 'kg$\cdot$m$^{-4}\cdot$s$^{-1}$'

        ax[0].plot(fvec, y)
        if scale_type == 'log' and newfig:
            ax[0].set_yscale('log')
        ax[1].plot(fvec, np.angle(zvec))
        
        if newfig:
            ax[0].set_ylabel(ylabel)
            ax[1].set_ylabel('phase (rad)')
            ax[1].set_xlabel('Frequency (Hz)')
        return ax


class Duct(PortImpedance):
    '''
    1-D duct object containing linear elements
    '''
    def __init__(self, world=None, losses=True):
        if not world:
            world = AcousticWorld()

        self.set_acoustic_world(world)
        self.elements = []
        self.element_positions = []
        self.termination = PerfectOpenEnd()
        self.world = world
        self.losses = losses

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
        element.set_parent(self)
        self.elements.append(element)
        self.update_element_pos()

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
        return z*self.char_impedance

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
        pos_el = zip(self.element_positions, self.elements)
        for order, (pos, el) in enumerate(pos_el):
            if pos+el.get_length() >= position:
                return order, el
        # out of bounds:
        return np.nan, None

    def get_radius_at_position(self, position=0.0):
        el_nbr, el = self.get_element_at_position(position=position)
        relative_pos = position - self.element_positions[el_nbr]
        el_rad = el.get_radius_at_position(relative_pos)
        return el_rad

    def get_total_length(self):
        return (self.element_positions[-1] + self.elements[-1].get_length())

    def normalized_transfer_mx_at_freq(self, freq=0.0,
                            from_pos=0.0, to_pos=None):
        """ Returns the pressure/ flow transfer matrix between
        two positions of the duct

        transfer_mx_at_freq(from_pos=0.0, to_pos=0.0)
        arguments:
            from_pos: position of source from upstream
            to_pos: position of source from upstream
            (negative position means relative to downstream)

        returns:
            2x2 matrix T such that:
                [pi, ui] = [t11,t12; t21, t22] [po, uo]
        """

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

        start_trans_mx = start_element.normalized_two_point_transfer_mx_at_freq
        if start_nb == end_nb:
            mx = start_trans_mx(from_pos=from_pos_rel,
                                to_pos=to_pos_rel,
                                freq=freq)
        else:
            mx = start_trans_mx(from_pos=from_pos_rel,
                                freq=freq)
            for el in self.elements[start_nb+1:end_nb]:
                mx = np.dot(mx, el.normalized_transfer_mx_at_freq(freq=freq))

            el = self.elements[end_nb]
            trans_mx = el.normalized_two_point_transfer_mx_at_freq
            mx = np.dot(mx, trans_mx(from_pos=0.0, to_pos=to_pos_rel,
                                     freq=freq))
        return mx

    def transfer_mx_at_freq(self, freq=0.0, from_pos=0.0, to_pos=None):
        mx = self.normalized_transfer_mx_at_freq(freq=freq,
                                                 from_pos=from_pos,
                                                 to_pos=to_pos)

        # remove normalization of the impedance
        mx[0, 1] *= self.char_impedance
        mx[1, 0] /= self.char_impedance
        return mx

    def travelling_mx_at_freq(self, freq=0.0,
                              from_pos=0.0, to_pos=None):
        """ Returns the travelling pressure transfer matrix between
        two positions of the duct

        transfer_mx_at_freq(from_pos=0.0, to_pos=0.0)
        arguments:
            from_pos: position of source from upstream
            to_pos: position of source from upstream
            (negative position means relative to downstream)

        returns:
            2x2 matrix T such that:
                [po1, pi1] = [t11,t12; t21, t22] [po2, pi2]
                (poX is the outgoing pressure wave at the "from" pos
                 piX the corresponding incoming wave)
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

    def plot_geometry(self, ax=None):
        if ax is None:
            newfig = True
            pl.figure()
            ax = pl.gca()
        else:
            newfig = False
        x, y = self.get_coords()
        ln = ax.plot(x, y)
        ax.plot(x, -np.array(y), color=ln[0].get_color())
        if newfig:
            ax.set_xlabel('distance from input (m)')
            ax.set_ylabel('radial distance (m)')

    def plot_report(self, ax=None, fmin=50.0, 
                    fmax=2000, npoints=200, scale_type='log'):
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
