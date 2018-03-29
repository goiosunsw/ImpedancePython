
import unittest
import numpy as np
import sys
import os
import scipy.io as sio

from pympedance.UNSW import *


script_path, _ = os.path.split(os.path.realpath(__file__))
print(script_path)
v6filename = os.path.join(script_path, 'data/A4_v6.mat')
v6parfile  = os.path.join(script_path, 'data/params.mat')
v7filename = os.path.join(script_path, 'data/euh.mat')
v7filelarge = os.path.join(script_path, 'data/i.mat')
v7file_infimp = os.path.join(script_path, 'data/InfImpCalib.mat')
v7file_infpipe = os.path.join(script_path, 'data/InfPipeCalib.mat')

# number of elements of large arrays to compare in asserts
n_comp = 10

import logging
logging.basicConfig(level=logging.INFO)

class UNSWTests(unittest.TestCase):
    def test_read_UNSW_v7_params(self):
        par = mat_parameter_from_file(v7filename)
        assert par is not None

    def test_read_UNSW_v6_params(self):
        par = mat_parameter_from_file(v6parfile)
        assert par is not None

    def test_read_UNSW_v6_file(self):
        self.assertRaises(KeyError,
                          mat_parameter_from_file,
                          v6filename)

    def test_UNSW_versionID_v6(self):
        iro = ImpedanceMeasurement(v6filename)
        assert iro.detectFormat(v6filename) == 'v6'

    def test_UNSW_versionID_v7(self):
        iro = ImpedanceMeasurement(v7filename)
        assert iro.detectFormat(v7filename) == 'v7'

    def test_param_consistency_v7(self):
        # param = mat_parameter_from_file(v7filename)
        pp = MeasurementParameters(v7filename)
        mdata = sio.loadmat(v7filename)
        f_old = mdata['Parameters'][0,0]['frequencyVector'].squeeze()
        f_new = pp.frequency_vector
        for fo,fn in zip(f_old, f_new):
            self.assertAlmostEqual(fo,fn)

    def test_iteration_v7(self):
        pp = MeasurementParameters(v7filename)
        mdata = sio.loadmat(v7filename)
        in_sig = mdata['Iteration'][0,0]['Input'][0,0]['originalWaveform']
        out_sig = mdata['Iteration'][0,0]['Output'][0,0]['waveform']
        ii = ImpedanceIteration(input_signals=in_sig,
                                  output_signal=out_sig,
                                  parameters=pp)

    def test_v7_recalc_calib(self):
        io = ImpedanceMeasurement(v7filename)
        old_a = io.parameters.A
        new_a = io.parameters.calc_calibration_matrix(infinite_imp_file=v7file_infimp,
                                                     infinite_pipe_file=v7file_infpipe)
        for old_dim, new_dim in zip(old_a.shape, new_a.shape):
            self.assertEqual(old_dim, new_dim)

        for ii in range(n_comp):
            ind = tuple((np.array(old_a.shape)*np.random.rand(3)).astype('i'))

            #self.assertAlmostEqual(old_a[ind],new_a[ind],places=3)

    def test_to_interpolated_impedance(self):
        io = ImpedanceMeasurement(v7filename)
        ti = io.as_interpolated_impedance()
        for ii in range(n_comp):
            ind = np.random.randint(len(io.f))
            f = io.f[ind]
            self.assertEqual(io.z[ind],ti._get_impedance_at_freq(f))

    def test_select_mics(self):

        io = ImpedanceMeasurement(v7filename)
        new_mics = (0,2)
        io2 = io.use_mics(new_mics)
        self.assertEqual(io2.mean_waveform.shape[1],2)
        par2=io2.parameters
        par=io.parameters
        self.assertEqual(len(par2.mic_pos),2)
        self.assertEqual(par2.A.shape[0],2)
        self.assertEqual(par2.A.shape[1],2)
        for new_i,old_i in enumerate(new_mics):
            self.assertEqual(par2.mic_pos[new_i],par.mic_pos[old_i])

    def test_detect_calibration_files(self):
        io = ImpedanceMeasurement(v7filename)
        self.assertIsNotNone(io.parameters.calib_files['inf_imp'])

    def test_v6_recalc_impedance(self):
        io = ImpedanceMeasurement(v6filename)
        missing_params = False
        try:
            rad = io.parameters.radius
        except AttributeError:
            missing_params = True
        imp = io.z
        try:
            new_imp = io.calculate_impedance()
        except ValueError:
            self.assertTrue(missing_params)
            return
        self.assertEqual(len(imp),len(new_imp))
        for zo, zn in zip(imp, new_imp):
            self.assertAlmostEqual(zo,zn)

    def test_v7_recalc_impedance(self):
        io = ImpedanceMeasurement(v7filename)
        imp = io.z
        new_imp = io.calculate_impedance()
        self.assertEqual(len(imp),len(new_imp))
        for zo, zn in zip(imp, new_imp):
            self.assertAlmostEqual(zo,zn, places=3)

    def test_v6_not_empty(self):
        io = ImpedanceMeasurement(v6filename)
        assert len(io.iterations) > 0

    def test_read_UNSW_v6(self):
        io = read_UNSW_impedance(v6filename)


    def test_read_UNSW_v7(self):
        io = read_UNSW_impedance(v7filename)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
