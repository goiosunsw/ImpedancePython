
import unittest
import numpy as np
import sys
import os

from ImpedanceUNSW import *


script_path, _ = os.path.split(os.path.realpath(__file__))
v6filename = os.path.join(script_path, 'data/A4_v6.mat')


class UNSWTests(unittest.TestCase):
    def test_UNSW_versionID_v6(self):
        iro = ImpedanceMeasurement(v6filename)

    def test_read_UNSW_v6(self):
        io = read_UNSW_impedance(v6filename)

def main():
    unittest.main()


if __name__ == '__main__':
    main()
