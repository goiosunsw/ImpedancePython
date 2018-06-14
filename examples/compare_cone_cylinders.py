import numpy as np
import matplotlib.pyplot as pl
import pympedance.Synthesiser as psy
from pympedance.plot_utils import bodeplot
from copy import copy
from collections import OrderedDict


l0 = 0.1
r0 = 0.01
term = psy.FlangedPiston(radius=r0)

obj_dict = OrderedDict()

cyls = psy.Duct()
cyls.append_element(psy.StraightDuct(length=l0,
                                     radius=r0))
cyls.set_termination(term)
obj_dict['cyl'] = cyls

f = np.array([10,100,1000])

r_out_list = [fac*r0 for fac in [1.1,1.2,1.5,2.0]]

for r_o in r_out_list:
    r_o
    cones = psy.Duct()
    cones.append_element(psy.ConicalDuct(length=l0, 
                                         radius_in=r0,
                                         radius_out=r_o))
    cones.set_termination(psy.FlangedPiston())
    obj_dict['cone %1.1f'%(r_o*100)] = cones

fig,ax = pl.subplots(2,sharex=True)

for k, obj in obj_dict.items():
    print('r: %f - %f'%(obj.get_radius_at_position(0),
                        obj.get_radius_at_position(obj.get_total_length())))
    print('z0: %f'%obj.elements[-1].char_impedance)
    print('z0norm: %f'%obj.elements[-1].normalized_impedance)
    print('r_l: %f (%f)'%(obj.termination.get_input_radius(),
                             obj.termination.radius))
    print('z0_l: %f'%obj.termination.char_impedance)
    print('z0norm_l: %f'%obj.termination.normalized_impedance)
    for ff in f:
        print('  z_l({}) = {}'.format(ff,obj.termination._get_impedance_at_freq(ff)))
        print('  z({}) = {}'.format(ff,obj.get_input_impedance_at_freq(np.array([ff]))))

    obj.plot_impedance(ax=ax,label=k)

pl.title('Cones with opening output')
pl.legend()

obj_dict = OrderedDict()
r_in = 0.01
r_out= 0.018
n_seg= 50
loss_mult=0.01

fig,ax = pl.subplots(2,sharex=True)
cyl = psy.Duct()
for ii in range(n_seg):
    l = l0/n_seg
    r = r_in + (r_out-r_in)*(ii+.5)/n_seg
    cyl.append_element(psy.StraightDuct(length=l,radius=r,loss_multiplier=loss_mult))
cyl.set_termination(psy.PerfectOpenEnd())

cyl.plot_impedance(ax=ax,label='step')
obj_dict['step %d'%n_seg] = cyl


#cone = psy.Duct()
#cone.append_element(psy.ConicalDuct(length=l0, 
#                                     radius_in=r_in,
#                                     radius_out=r_out))
#cone.set_termination(psy.FlangedPiston())
#obj_dict['cone'] = cone

#cone.plot_impedance(ax=ax,label='cone')

# pl.legend()

#fig,ax = pl.subplots(2,sharex=True)
#r_in=0.01
#r_out = 0.011

cone = psy.Duct()
cone.append_element(psy.ConicalDuct(length=l0, 
                                     radius_in=r_in,
                                     radius_out=r_out,loss_multiplier=loss_mult))
cone.set_termination(psy.PerfectOpenEnd())

# impedance according to scavone
f = np.logspace(0,3.5,1000)



z_sim = cone.get_input_impedance_at_freq(f)

bodeplot(f,z_sim,ax=ax,label='simulated')

c = cone.elements[-1].get_speed_of_sound()
rho = cone.elements[-1].get_medium_density()
y0 = np.pi*r_in**2/rho/c
x_apex = l0/(r_out-r_in)*r_in
phase_apex = 2*np.pi*f/c * x_apex
eph = np.exp(1j*phase_apex)
phase = 2*np.pi*f/c * (l0+x_apex) * 2
r = -1 * np.exp(-1j*phase)
z = 1/y0/((1/eph-r*eph)/(1/eph+r*eph)+1/1j/phase_apex)

bodeplot(f,z,ax=ax,label='theo' )
pl.legend()


pl.show()
