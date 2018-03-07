
import unittest
import numpy as np
import sys
import os

from pympedance.UNSWreader import *


script_path, _ = os.path.split(os.path.realpath(__file__))
v6filename = os.path.join(script_path, 'data/A4_v6.mat')
v7filename = os.path.join(script_path, 'data/euh.mat')

class UNSWTests(unittest.TestCase):
    def test_UNSW_versionID_v6(self):
        iro = ImpedanceMeasurement(v6filename)
        assert iro.detectFormat(v6filename) == 'v6'
    
    def test_UNSW_versionID_v7(self):
        iro = ImpedanceMeasurement(v7filename)
        assert iro.detectFormat(v7filename) == 'v7'

    def test_read_UNSW_v6(self):
        io = read_UNSW_impedance(v6filename)

    def test_read_UNSW_v7(self):
        io = read_UNSW_impedance(v7filename)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
