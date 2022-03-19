import numpy as np
import pandas as pd
import pandas.testing
from panndas import nn
from panndas.data import utils


class TestReLU:
    def test_relu_negative(self):
        """Tests whether ReLU is 0. everywhere if given all negative values."""
        array = -1.0 * np.ones((4, 3))
        zeroes = np.zeros_like(array)

        df = pd.DataFrame(data=array)
        zero_df = pd.DataFrame(data=zeroes)

        r = nn.ReLU()

        pandas.testing.assert_frame_equal(r(df), zero_df)

    def test_relu_nonnegative(self):
        """Tests whether ReLU is identity if given all non-negative values."""
        eye = utils.eye(range(10))
        r = nn.ReLU()

        pandas.testing.assert_frame_equal(r(eye), eye)


class TestLinearAttention:
    def test_identity(self):
        """Tests whether the LinearAttention module with identity weights is an identity."""
        eye = utils.eye(list(range(5)))
        df = pd.DataFrame(data=eye)

        a = nn.LinearAttention(df, df, df)

        pandas.testing.assert_frame_equal(a(df), df)


class TestSoftmaxAttention:
    def test_identity(self):
        """Tests the SoftmaxAttention module with identity as weights."""
        eye = utils.eye(list(range(5)))
        df = pd.DataFrame(data=eye)

        a = nn.SoftmaxAttention(df, df, df)

        expected_data = (eye * 0.2558) + (0.1488 * np.ones_like(eye))
        expected = pd.DataFrame(
            data=expected_data, index=eye.index, columns=eye.columns
        )

        pandas.testing.assert_frame_equal(a(df), expected, check_exact=False, rtol=1e-3)
