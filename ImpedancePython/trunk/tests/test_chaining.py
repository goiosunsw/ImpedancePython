import unittest
import numpy as np
import sys

from ImpedanceSynthesiser import *


def freq_vector(f_st = 100, f_end = 10000, n = 50, log=False):
    if log:
        return np.logspace(np.log10(f_st), np.log10(f_end), n)
    else:
        return np.linspace(f_st, f_end, n)


def random_duct(n_segments=None, lengths=[], radii=[]):
    if n_segments is None:
        n_segments = 1+int(9*np.random.rand())

    duct = Duct()
    for seg_nbr in range(n_segments):
        try:
            l = lengths[seg_nbr]
        except IndexError:
            l = np.random.rand()
        try:
            r = radii[seg_nbr]
        except IndexError:
            r = np.random.rand()

        duct.append_element(StraightDuct(length=l, radius=r))

    duct.set_termination(PerfectOpenEnd())

    return duct


class WorldTests(unittest.TestCase):
    def test_default_world(self):
        world = AcousticWorld()
        print('')
        print('c = {} m/s^2'.format(world.speed_of_sound))
        print('rho = {} kg/m^3'.format(world.medium_density))

    def test_compare_factors(self):
        fact = [(25,0),(25,0.5),(25,1.0),(0,0),(0,0.5),(0,1.0)]

        print('')
        for t, h in fact:
            world = AcousticWorld(temp=t, humid=h)
            print('temp = {} ; humid = {}'.format(t, h))
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

    def test_chaining_in_single_element(self):
        sec_len = 1
        mid_pos=.4
        rad0 = 1

        duct = Duct()
        duct.set_termination(PerfectOpenEnd())
        this_duct = StraightDuct(length=sec_len,radius=rad0)
        duct.append_element(this_duct)

        fvec = freq_vector()

        err_msg = 'Failed at freq {}:\n * tm_c[{},{}] = {},\n   expected {}'

        for f in fvec:
            tm_all = duct.transfer_mx_at_freq(freq=f)

            tm1 = duct.transfer_mx_at_freq(freq=f,
                                           from_pos=0.0,
                                           to_pos=mid_pos)
            tm2 = duct.transfer_mx_at_freq(freq=f,
                                           from_pos=mid_pos)
            tm_comp = np.dot(tm1, tm2)


            for row in range(tm_comp.shape[0]):
                for col in range(tm_comp.shape[1]):
                    self.assertAlmostEqual(tm_all[row, col], tm_comp[row, col],
                                          msg=err_msg.format(f,row,col,
                                                             tm_comp[row,col],
                                                             tm_all[row,col]))

    def test_duct_transfer_mx(self):
        sec_len = 1
        mid_pos=.4
        rad0 = 1

        duct = Duct()
        duct.set_termination(PerfectOpenEnd())
        this_duct = StraightDuct(length=sec_len,radius=rad0)
        duct.append_element(this_duct)

        fvec = freq_vector()

        err_msg = 'Failed at freq {}:\n * tm_s[{},{}] = {},\n   expected {}'

        for f in fvec:
            tm_d = duct.transfer_mx_at_freq(freq=f)

            tm_s = this_duct.transfer_mx_at_freq(freq=f)
            for row in range(tm_d.shape[0]):
                for col in range(tm_d.shape[1]):
                    self.assertAlmostEqual(tm_d[row, col], tm_s[row, col],
                                           msg=err_msg.format(f, row, col,
                                                              tm_d[row, col],
                                                              tm_s[row, col]))

    def test_travelling_mx_in_single_section(self):
        duct = random_duct(n_segments=1)
        section = duct.elements[0]

        fvec = freq_vector()

        err_msg = 'Failed at freq {}:\n * tm_s[{},{}] = {},\n   expected {}'

        for f in fvec:
            tm_d = duct.travelling_mx_at_freq(freq=f)
            tm_s = section.travelling_mx_at_freq(freq=f)
            for row in range(tm_d.shape[0]):
                for col in range(tm_d.shape[1]):
                    self.assertAlmostEqual(tm_d[row, col], tm_s[row, col],
                                           msg=err_msg.format(f, row, col,
                                                              tm_d[row, col],
                                                              tm_s[row, col]))

    def test_travelling_mx_in_chained_sections(self):
        n_seg = 1+np.random.randint(9)
        radii = np.random.random() * np.ones(n_seg)
        duct = random_duct(n_segments=n_seg, radii=radii)
        section = StraightDuct(length=duct.get_total_length(), radius=radii[0])

        fvec = freq_vector()

        err_msg = 'Failed at freq {}:\n * tm_s[{},{}] = {},\n   expected {}'

        for f in fvec:
            tm_d = duct.travelling_mx_at_freq(freq=f)
            tm_s = section.travelling_mx_at_freq(freq=f)
            for row in range(tm_d.shape[0]):
                for col in range(tm_d.shape[1]):
                    self.assertAlmostEqual(tm_d[row, col], tm_s[row, col],
                                           msg=err_msg.format(f, row, col,
                                                              tm_d[row, col],
                                                              tm_s[row, col]))

    def test_transfer_mx_in_single_section(self):
        duct = random_duct(n_segments=1)
        section = duct.elements[0]

        fvec = freq_vector()

        err_msg = 'Failed at freq {}:\n * tm_s[{},{}] = {},\n   expected {}'

        for f in fvec:
            tm_d = duct.transfer_mx_at_freq(freq=f)
            tm_s = section.transfer_mx_at_freq(freq=f)
            for row in range(tm_d.shape[0]):
                for col in range(tm_d.shape[1]):
                    self.assertAlmostEqual(tm_d[row, col], tm_s[row, col],
                                           msg=err_msg.format(f, row, col,
                                                              tm_d[row, col],
                                                              tm_s[row, col]))

    def test_transfer_mx_in_chained_sections(self):
        n_seg = 2 #1+np.random.randint(2)
        radii = np.random.random() * np.ones(n_seg)
        duct = random_duct(n_segments=n_seg, radii=radii)
        section = StraightDuct(length=duct.get_total_length())

        fvec = freq_vector()

        err_msg = 'Failed at freq {}:\n * tm_s[{},{}] = {},\n   expected {}'

        for f in fvec:
            tm_d = duct.transfer_mx_at_freq(freq=f)
            tm_s = section.transfer_mx_at_freq(freq=f)
            for row in range(tm_d.shape[0]):
                for col in range(tm_d.shape[1]):
                    self.assertAlmostEqual(tm_d[row, col], tm_s[row, col],
                                           msg=err_msg.format(f, row, col,
                                                              tm_d[row, col],
                                                              tm_s[row, col]))

    def test_position_out_of_duct(self):
        duct = random_duct(n_segments=1)
        total_length = duct.get_total_length()
        el_nbr, el = duct.get_element_at_position(total_length+1)
        self.assertIsNone(el)
        self.assertTrue(np.isnan(el_nbr))

    def test_position_at_edge_of_duct(self):

        duct = random_duct()
        total_nbr = len(duct.elements)
        total_length = duct.get_total_length()
        el_nbr, el = duct.get_element_at_position(total_length)
        self.assertEqual(el_nbr, total_nbr-1)

    def test_total_length(self):
        duct = random_duct()
        total_length = duct.get_total_length()
        len_sum = 0
        for el in duct.elements:
            len_sum += el.get_length()

        self.assertEqual(total_length, len_sum)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
