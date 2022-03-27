import numpy as np
import pandas as pd
import pandas.testing
from panndas import nn


class TestLayerMaxNorm:
    def test_import_show(self):
        """Tests whether a LayerMaxNorm Module can be imported and shown."""
        n = nn.LayerMaxNorm()
        show = n.show()
        assert "LayerMaxNorm" in show

    def test_forward(self):
        """Tests whether the forward method of LayerMaxNorm Module works."""
        rows, cols = 5, 10
        xs = pd.DataFrame(data=2 * np.random.rand(rows, cols))
        maxidxs = xs.idxmax(axis=0)

        n = nn.LayerMaxNorm()

        outs = n(xs)  # testing behavior on positive numbers

        assert all(outs.max(axis=0) <= 1.0)
        desired_maxs = pd.Series([1.0] * cols)
        maxs_in_outs = pd.Series(
            outs.iloc[maxidxs.loc[col], ii] for ii, col in enumerate(outs.columns)
        )
        pandas.testing.assert_series_equal(
            maxs_in_outs, desired_maxs, check_exact=False
        )

        outs = n(-1.0 * xs)  # testing behavior on negative numbers

        assert all(outs.min(axis=0) >= -1.0)
        desired_mins = pd.Series([-1.0] * cols)
        mins_in_outs = pd.Series(
            outs.iloc[maxidxs.loc[col], ii] for ii, col in enumerate(outs.columns)
        )
        pandas.testing.assert_series_equal(
            mins_in_outs, desired_mins, check_exact=False
        )


class TestBatchNorm:
    def test_import_show(self):
        """Tests whether a BatchNorm1d Module can be imported and shown."""
        d = nn.BatchNorm1d()
        show = d.show()
        assert "BatchNorm1d" in show

    def test_forward(self):
        """Tests whether the forward method of BatchNorm1d Module works."""
        rows, cols = 5, 10
        xs = pd.DataFrame(data=2 * np.random.rand(rows, cols) - 1.0)
        gamma, beta = np.sqrt(2), 1.0

        n = nn.BatchNorm1d(eps=0.0, gamma=gamma, beta=beta)
        outs = n(xs)

        desired_means = pd.Series([beta] * rows)
        out_means = outs.mean(axis=1)
        pandas.testing.assert_series_equal(out_means, desired_means, check_exact=False)

        desired_stds = pd.Series([gamma] * rows)
        out_stds = outs.std(ddof=0, axis=1)
        pandas.testing.assert_series_equal(out_stds, desired_stds, check_exact=False)
