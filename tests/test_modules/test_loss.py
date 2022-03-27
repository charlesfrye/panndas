import pandas as pd
import pandas.testing
from panndas import nn
from panndas.data import utils


class TestLp:
    def test_import_show(self):
        """Tests whether an Lp loss can be imported and shown."""
        cnames = pd.Index(["ayy", "lmao"], name="whats good")
        inames = pd.Index([1, 2], name="buckle my shoe")
        df = pd.DataFrame(columns=cnames, index=inames)
        m = nn.Linear(df)
        show = m.show()
        assert "Linear" in show
        assert cnames.name in show
        assert inames.name in show

    def test_forward_eye(self):
        """Tests whether a Linear module with the eye-dentity weight matrix is an identity."""
        eye = utils.eye(list(range(5)))
        df = pd.DataFrame(data=eye)
        m = nn.Linear(df, bias_series=0.0)
        pandas.testing.assert_frame_equal(m(df), df)

    def test_bias_series(self):
        """Tests whether the bias_series of a Linear module is set correctly."""
        eye = utils.eye(list(range(5)))
        df = pd.DataFrame(data=eye)
        m_float = nn.Linear(df, bias_series=0.0)
        m_series = nn.Linear(df, bias_series=m_float.b)

        assert m_float.b.map(lambda b: b == 0.0).all()
        pandas.testing.assert_series_equal(m_float.b, m_series.b)
