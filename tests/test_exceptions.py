''' Script to check pylandau exceptions.
'''
import pytest

import numpy as np
import pylandau

class TestExceptions():

    def test_negative_amplitude(self):
        ''' Check exception for negative amplitude '''
        x = np.linspace(0, 100)
        with pytest.raises(ValueError):
            pylandau.landau(x, A=-1)

        x = np.linspace(0, 100)
        with pytest.raises(ValueError):
            pylandau.langau(x, A=-1)

if __name__ == '__main__':
    pytest.main()
