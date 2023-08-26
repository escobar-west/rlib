import polars as pl
import numpy as np
from scipy import special


def ref_pricer(df: pl.DataFrame, rate: float) -> pl.DataFrame:
    rate = pl.lit(rate)
    var_factor = df["sigma"] * df["maturity"].sqrt()
    d1 = (
        df["maturity"] * (rate + df["sigma"] ** 2 / 2)
        + (df["asset_price"] / df["strike"]).log()
    ) / var_factor
    d2 = d1 - var_factor
    df = df.with_columns(
        option_price=norm_cdf(d1) * pl.col("asset_price")
        - norm_cdf(d2) * pl.col("strike") * (-rate * pl.col("maturity")).exp()
    )

    return df


def norm_cdf(x: pl.Series):
    return (1 + special.erf(x / np.sqrt(2))) / 2
