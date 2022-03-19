from panndas import nn
import pytest


class TestModule:
    def test_import_show(self):
        """Tests whether the base Module can be imported and shown."""
        m = nn.Module()
        assert m.show() == "Module"

    def test_forward_raises(self):
        """Tests that a base Module cannot be called."""
        m = nn.Module()
        with pytest.raises(NotImplementedError):
            m(None)
