''' Script to check pylandau exceptions.
'''
import unittest

import numpy as np
import pylandau


class TestExceptions(unittest.TestCase):

    def test_negative_amplitude(self):
        ''' Check exception for negative amplitude '''
        x = np.linspace(0, 100)
        with self.assertRaises(ValueError):
            pylandau.landau(x, A=-1)

        x = np.linspace(0, 100)
        with self.assertRaises(ValueError):
            pylandau.langau(x, A=-1)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestExceptions)
    unittest.TextTestRunner(verbosity=2).run(suite)
