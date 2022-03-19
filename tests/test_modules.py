import pandas as pd
import pandas.testing
from panndas import nn
import panndas.nn.math
import pytest


def test_unit():  # unit, in the monoidal sense
    assert True


def test_module_import_show():
    m = nn.Module()
    assert m.show() == "Module"


def test_module_forward_raises():
    m = nn.Module()
    with pytest.raises(NotImplementedError):
        m(None)


def test_linear_import_show():
    cnames = pd.Index(["ayy", "lmao"], name="whats good")
    inames = pd.Index([1, 2], name="buckle my shoe")
    df = pd.DataFrame(columns=cnames, index=inames)
    m = nn.Linear(df)
    show = m.show()
    assert "Linear" in show
    assert cnames.name in show
    assert inames.name in show


def test_linear_forward_eye():
    eye = panndas.nn.math.eye(list(range(5)))
    df = pd.DataFrame(data=eye)
    m = nn.Linear(df, bias_series=0.0)
    assert m(df).equals(df)


def test_sequential_list():
    ms = [nn.Module() for _ in range(5)]
    s = nn.Sequential(ms)

    assert ms[0] in s  # test __iter__
    assert s[3] == ms[3]  # test __getitem__
    assert s.modules == ms  # test __init__

    with pytest.raises(NotImplementedError):
        s(None)  # test forward


def test_sequential_show():
    ms = [nn.Module() for _ in range(5)]
    s = nn.Sequential(ms)

    show = s.show()

    assert "Sequential" in show
    assert "Module" in show
    assert len(show.split(",")) == len(s.modules)


def test_linear_bias_series():
    eye = panndas.nn.math.eye(list(range(5)))
    df = pd.DataFrame(data=eye)
    m_float = nn.Linear(df, bias_series=0.0)
    m_series = nn.Linear(df, bias_series=m_float.b)

    assert m_float.b.map(lambda b: b == 0.0).all()
    pandas.testing.assert_series_equal(m_float.b, m_series.b)


def test_sequential_linear_eye():
    eye = panndas.nn.math.eye(list(range(5)))
    df = pd.DataFrame(data=eye)
    ms = [nn.Linear(df, bias_series=0.0) for _ in range(3)]
    s = nn.Sequential(ms)
    assert s(df).equals(df)


def test_additive_skip():
    m = nn.Module()
    a = nn.AdditiveSkip(m)

    show = a.show()
    m_show = m.show()

    assert a.block == m
    assert "AdditiveSkip" in show
    assert m_show in show


def test_skip_linear():
    eye = panndas.nn.math.eye(list(range(7)))
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
