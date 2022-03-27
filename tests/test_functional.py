"""Test functional interface by comparing with PyTorch."""
import numpy as np
import pandas as pd
import panndas.nn.functional as F
import torch
import torch.nn.functional as tF


class TestFunctional:
    """Tests components of the functional module on DataFrames and Series.

    Includes convenience methods for generating data and for testing almost equality
    of pandas data and torch tensors."""

    # testing dataframe methods
    df = pd.DataFrame(data=np.arange(-30.0, 30.0, step=1).reshape(12, 5))
    t = torch.Tensor(df.to_numpy())

    # testing series methods
    s = df.stack()
    t_flat = torch.Tensor(s.to_numpy())

    def assert_allclose(self, pdata, tensor, rtol=1e-7):
        np.testing.assert_allclose(pdata.to_numpy(), tensor.numpy(), rtol=rtol)


class TestPointwise(TestFunctional):
    """Tests pointwise operations on DataFrames and Series."""

    def test_relu(self):
        self.assert_allclose(F.relu(self.df), tF.relu(self.t))
        self.assert_allclose(F.relu(self.s), tF.relu(self.t_flat))

    def test_tanh(self):
        self.assert_allclose(F.tanh(self.df), torch.tanh(self.t))
        self.assert_allclose(F.tanh(self.s), torch.tanh(self.t_flat))

    def test_softplus(self):
        self.assert_allclose(F.softplus(self.df), tF.softplus(self.t), rtol=2e-3)
        self.assert_allclose(F.softplus(self.s), tF.softplus(self.t_flat), rtol=2e-3)

    def test_mish(self):
        self.assert_allclose(F.mish(self.df), tF.mish(self.t), rtol=2e-3)
        self.assert_allclose(F.mish(self.s), tF.mish(self.t_flat), rtol=2e-3)

    def test_sigmoid(self):
        self.assert_allclose(F.sigmoid(self.df), torch.sigmoid(self.t))
        self.assert_allclose(F.sigmoid(self.s), torch.sigmoid(self.t_flat))

    def test_softmax(self):
        self.assert_allclose(F.softmax(self.df), tF.softmax(self.t, dim=0), rtol=1e-6)
        self.assert_allclose(
            F.softmax(self.s), tF.softmax(self.t_flat, dim=0), rtol=1e-6
        )
