from panndas import nn
import pytest


class TestSequential:
    def test_sequential_list(self):
        """Tests the interface for a Sequential module."""
        ms = [nn.Module() for _ in range(5)]
        s = nn.Sequential(ms)

        assert ms[0] in s  # test __iter__
        assert s[3] == ms[3]  # test __getitem__
        assert s.modules == ms  # test __init__

        with pytest.raises(NotImplementedError):
            s(None)  # test forward

    def test_sequential_show(self):
        """Tests whether a Sequential module can be shown."""
        ms = [nn.Module() for _ in range(5)]
        s = nn.Sequential(ms)

        show = s.show()

        assert "Sequential" in show
        assert "Module" in show
        assert len(show.split(",")) == len(s.modules)

    def test_nested_sequential_show(self):
        """Tests whether nested Sequentials display properly."""
        ms = [nn.Module() for _ in range(5)]
        seq1 = nn.Sequential(ms)

        ns = [nn.Module() for _ in range(2)]
        seq2 = nn.Sequential(ns)

        s = nn.Sequential([seq1, seq2])

        show = s.show()

        assert "Sequential" in show
        assert "Module" in show
        assert "\\'" not in show
