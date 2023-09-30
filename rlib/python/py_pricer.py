import polars as pl
import numpy as np
from scipy import special

__all__ = ["py_ref_pricer", "py_mc_pricer"]


def py_ref_pricer(df: pl.DataFrame, rate: float) -> pl.DataFrame:
    rate = pl.lit(rate)
    var_factor = pl.col("sigma") * pl.col("maturity").sqrt()
    d1 = (
        pl.col("maturity") * (rate + 0.5 * pl.col("sigma") ** 2)
        + (pl.col("asset_price") / pl.col("strike")).log()
    ) / var_factor
    d2 = d1 - var_factor
    df = df.with_columns(
        option_price=norm_cdf(d1) * pl.col("asset_price")
        - norm_cdf(d2) * pl.col("strike") * (-rate * pl.col("maturity")).exp()
    )
    return df


def py_mc_pricer(df: pl.DataFrame, rate: float, n_sims: int) -> pl.DataFrame:
    sigma = df["sigma"].view().reshape(-1, 1)
    maturity = df["maturity"].view().reshape(-1, 1)
    strike = df["strike"].view().reshape(-1, 1)
    asset_price = df["asset_price"].view().reshape(-1, 1)

    Z = np.random.normal(size=(df.height, n_sims))
    rand_walk = sigma * np.sqrt(maturity) * Z
    mean_drift = (rate - 0.5 * sigma**2) * maturity
    paths = asset_price * np.exp(mean_drift + rand_walk)
    exp_payoff = (np.exp(-rate * maturity) * np.maximum(paths - strike, 0)).mean(axis=1)
    df = df.with_columns(option_price=pl.Series(exp_payoff))
    return df


def norm_cdf(x: pl.Series):
    return 0.5 * (1 + special.erf(x / np.sqrt(2)))
