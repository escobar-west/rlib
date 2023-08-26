import polars as pl
from polars.testing import assert_series_equal
from rlib.py_pricer import ref_pricer


def test_ref_pricer():
    input = pl.DataFrame(
        {"strike": [100], "asset_price": [102], "sigma": [0.2], "maturity": [0.5]}
    )
    output = ref_pricer(input, 0.02)
    reference = pl.Series("option_price", [7.288151])
    assert_series_equal(output["option_price"], reference)
