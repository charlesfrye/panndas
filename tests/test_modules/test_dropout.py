import numpy as np
import pandas as pd
from panndas import nn


class TestAlphaDropout:
    def test_import_show(self):
        """Tests whether an AlphaDropout module can be imported and shown."""
        p, alpha = 0.5, 0.1
        d = nn.AlphaDropout(p=p, alpha=alpha)
        show = d.show()
        assert "AlphaDropout" in show
        assert str(p) in show
        assert str(alpha) in show

    def test_forward(self):
        """Tests whether the fraction of dropped values is correct.

        This test has a flake probability of approximately one in 2,000,000.

        Bayes' rule would tell us that any reasonable prior on bugs in code
        will result in a high positive predictive value for this test's failure.
        """
        size = 100
        xs = pd.DataFrame(data=np.random.rand(size, size))
        p = 0.25
        d = nn.AlphaDropout(p=p, alpha=np.nan)
        outs = d(xs)
        assert type(outs) == type(xs)
        masked_p = outs.isna().sum().sum() / (size**2)
        sd = np.sqrt(p * (1 - p)) / size
        lower, upper = (p - 5 * sd, p + 5 * sd)
        assert lower < masked_p < upper


class TestDropout:
    def test_import_show(self):
        """Tests whether a Dropout module can be imported and shown."""
        p = 0.5
        d = nn.Dropout(p=p)
        show = d.show()
        assert "Dropout" in show
        assert str(p) in show
