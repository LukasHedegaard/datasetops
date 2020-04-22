from .utils import from_dummy_data
from datasetops import filters
import pytest


def test_filter():
    num_total = 10
    ds = from_dummy_data(num_total=num_total, with_label=True).named("data", "label")

    # expected items
    a = [(x, str("a")) for x in list(range(5))]
    b = [(x, str("b")) for x in list(range(5, num_total))]
    even_a = [x for x in a if x[0] % 2 == 0]
    even_b = [x for x in b if x[0] % 2 == 0]

    # element wise
    assert (
        even_a + even_b
        == list(ds.filter("data", lambda x: x % 2 == 0))
        == list(ds.filter(0, lambda x: x % 2 == 0))
    )

    # bulk
    assert list(ds.filter(lambda s: s[0] % 2 == 0 and s[1] == "a")) == even_a

    # sample_classwise
    ds_classwise = ds.filter("label", filters.allow_unique(2))
    assert list(ds_classwise) == list(a[:2] + b[:2])

    # error scenarios
    with pytest.raises(KeyError):
        ds.filter("badkey", lambda x: True)  # key doesn't exist


def test_allow_unique():
    num_total = 10
    ds = from_dummy_data(num_total=num_total, with_label=True).named("data", "label")

    # expected items
    a = [(x, str("a")) for x in list(range(5))]
    b = [(x, str("b")) for x in list(range(5, num_total))]

    ds_classwise = ds.filter("label", filters.allow_unique(2))
    assert list(ds_classwise) == list(a[:2] + b[:2])
