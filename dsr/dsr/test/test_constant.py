from dsr.library import HardCodedConstant
import numpy as np

def test_constant():
    valid_cases = np.arange(0, 25, 0.1)
    for number in valid_cases:
        const = HardCodedConstant(value=number)
        assert const() == number, "Value returned from Constant.function() ({}) does not match input value ({}).".format(const(), number)
