import pandas as pd
import pandas.testing
from panndas import nn
from panndas.data import utils


class AdditiveSkip:
    def test_import_show(self):
        """Tests whether the AdditiveSkip module can be imported and shown."""
        m = nn.Module()
        a = nn.AdditiveSkip(m)

        show = a.show()
        m_show = m.show()

        assert a.block == m
        assert "AdditiveSkip" in show
        assert m_show in show

    def test_skip_linear(self):
        """Tests whether the AdditiveSkip module works correctly with a Linear module."""
        eye = utils.eye(list(range(7)))
        df = pd.DataFrame(data=eye)
        m = nn.Linear(eye, bias_series=0.0)

        a = nn.AdditiveSkip(m)

        show = a.show()
        m_show = m.show()

        assert a.block == m
        assert "AdditiveSkip" in show
        assert m_show in show

        out = a(df)
        pandas.testing.assert_frame_equal(out, df + df)
