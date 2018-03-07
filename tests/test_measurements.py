import unittest
import numpy as np
import sys

from pympedance.Synthesiser import *
from pympedance.Measurement import *
from pympedance import Impedance



class MeasurementTests(unittest.TestCase):
    def test_head_checks_short_duct(self):
        l=.1
        r=.05
        duct = Duct()
        duct.append_element(StraightDuct(length=l,radius=r))
        sens = SensorList()
        sens.set_positions([l/2,2*l])
        head = ImpedanceHead(duct=duct,sensor_set=sens)
        self.assertFalse(head.check_sensor_consistency())

    def test_head_checks_suitable_duct(self):
        head = ImpedanceHead()
        l = head.base_geometry.get_total_length()
        #sens = SensorList()
        #sens.set_positions([l/4,l/2])
        head.sensor_set.set_positions([l/4,l/2])
        self.assertTrue(head.check_sensor_consistency())

def main():
    unittest.main()


if __name__ == '__main__':
    main()
