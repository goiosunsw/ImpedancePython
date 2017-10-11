import unittest
import numpy as np

from ImpedanceSynthesiser import *

def freq_vector(f_st = 100, f_end = 10000, n = 1000, log=False):
    if log:
        return np.logspace(np.log10(f_st),np.log10(f_end),n)
    else:
        return np.linspace(f_st,f_end,n)

class WorldTests(unittest.TestCase):
    def test_default_world(self):
        world = AcousticWorld()
        print('')
        print('c = {} m/s^2'.format(world.speed_of_sound))
        print('rho = {} kg/m^3'.format(world.medium_density))
        
    def test_compare_factors(self):
        fact = [(25,0),(25,0.5),(25,1.0),(0,0),(0,0.5),(0,1.0)]
        
        print('')
        for t,h in fact:
            world = AcousticWorld(temp=t, humid=h)
            print('temp = {} ; humid = {}'.format(t,h))
            print('c = {} m/s^2'.format(world.speed_of_sound))
            print('rho = {} kg/m^3'.format(world.medium_density))

class TypeTests(unittest.TestCase):
    def test_open_end_is_termination_impedance(self):
        term = PerfectOpenEnd()
        self.assertIsInstance(term, TerminationImpedance)
        
    def test_closed_end_is_termination_impedance(self):
        term = PerfectClosedEnd()
        self.assertIsInstance(term, TerminationImpedance)

class ChainingTests(unittest.TestCase):
    
    def test_default_section_plus_termination(self):
        mid = DuctSection()
        term = TerminationImpedance()
        
        fvec = freq_vector()
        
        for f in fvec:
            r = term._get_reflection_coeff_at_freq(f)
            r = mid._chain_reflection_coeff_at_freq(r, f)
            self.assertEqual(r,-1.)
        
    def test_default_section_plus_closed_end(self):
        mid = DuctSection()
        term = PerfectClosedEnd()
        
        fvec = freq_vector()
        
        for f in fvec:
            r = term._get_reflection_coeff_at_freq(f)
            r = mid._chain_reflection_coeff_at_freq(r, f)
            self.assertEqual(r,1.)
        
    def test_two_default_sections_plus_closed_end(self):
        mid1 = DuctSection()
        mid2 = DuctSection()
        term = PerfectClosedEnd()
        
        fvec = freq_vector()
        
        for f in fvec:
            r = term._get_reflection_coeff_at_freq(f)
            r = mid1._chain_reflection_coeff_at_freq(r, f)
            r = mid2._chain_reflection_coeff_at_freq(r, f)
            self.assertEqual(r,1.)
            
    def test_impedance_closed_end(self):
        term = PerfectClosedEnd()
        
        fvec = freq_vector()
        
        for f in fvec:
            z = term._get_impedance_at_freq(f)
            self.assertEqual(z,np.inf)

    def test_impedance_two_default_sections_plus_closed_end(self):
        mid1 = DuctSection()
        mid2 = DuctSection()
        term = PerfectClosedEnd()
        
        fvec = freq_vector()
        
        for f in fvec:
            z = term._get_impedance_at_freq(f)
            z = mid1._chain_impedance_at_freq(z, f)
            z = mid2._chain_impedance_at_freq(z, f)
            self.assertEqual(z,np.inf)
            
    def test_impedance_two_default_sections_plus_open_end(self):
        mid1 = DuctSection()
        mid2 = DuctSection()
        term = PerfectOpenEnd()
        
        fvec = freq_vector()
        
        for f in fvec:
            z = term._get_impedance_at_freq(f)
            z = mid1._chain_impedance_at_freq(z, f)
            z = mid2._chain_impedance_at_freq(z, f)
            self.assertEqual(z,0)
            
    def test_one_straight_tube_plus_closed_end(self):
        mid1 = StraightDuct()
        term = PerfectClosedEnd()
        
        fvec = freq_vector()
        
        for f in fvec:
            r = term._get_reflection_coeff_at_freq(f)
            r = mid1._chain_reflection_coeff_at_freq(r, f)
            self.assertAlmostEqual(np.abs(r),1.)

class DuctTests(unittest.TestCase):
    
    def test_rf_default_section_plus_termination(self):
        
        duct = Duct()
        duct.set_termination(PerfectOpenEnd())
        duct.append_element(DuctSection())
        
        fvec = freq_vector()
        
        for f in fvec:
            r = duct.get_input_reflection_function_at_freq(f)
            self.assertEqual(r,-1.)
    
   
    def test_default_section_plus_termination(self):
        
        duct = Duct()
        duct.set_termination(PerfectOpenEnd())
        duct.append_element(DuctSection())
        
        fvec = freq_vector()
        
        for f in fvec:
            z = duct.get_input_impedance_at_freq(f)
            self.assertEqual(z,0.)

    def test_cylinder_plus_termination(self):
        
        duct = Duct(losses=False)
        duct.set_termination(PerfectOpenEnd())
        mid1 = StraightDuct()
        duct.append_element(mid1)
        
        fvec = freq_vector()
        l = mid1.length
        c = mid1.get_speed_of_sound()
        z0 = mid1.get_characteristic_impedance()
        
        for f in fvec:
            z_exp = 1j*z0*np.tan(2*np.pi*f*l/c)
            z = duct.get_input_impedance_at_freq(f)
            self.assertAlmostEqual(z,z_exp,
               msg='Failed at freq {}: z={}, expected {}'.format(f,z,z_exp))

    def test_two_straight_element_coords(self):
        
        sec_len = 1
        rad0 = 1
        rad1 = 2
        lengths = [sec_len,sec_len]
        radiuses = [rad0,rad1]
        x = [0,sec_len,sec_len,sec_len*2]
        y = [rad0,rad0,rad1,rad1]
        
        duct = Duct()
        duct.set_termination(PerfectOpenEnd())
        mid = []
        for l,r in zip(lengths, radiuses):
            this_duct = StraightDuct(length=l,radius=r)
            duct.append_element(this_duct)
            mid.append(this_duct)
        
        xc,yc = duct.get_coords()
        self.assertListEqual(xc,x)
        self.assertListEqual(yc,y)
        
        
        duct.get_coords()
        

def main():
    unittest.main()

if __name__ == '__main__':
    main()
