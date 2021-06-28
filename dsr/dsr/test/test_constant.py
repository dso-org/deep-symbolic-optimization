from dsr.library import Constant
import pytest
import numpy as np

def test_constant():
    valid_cases = np.arange(0, 25, 0.1)
    for number in valid_cases:
        const = Constant(value=number)
        assert const.function() == number, f"Value returned from Constant.function() {const.function()} does not match input value {number}."