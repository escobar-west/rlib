# Rlib

POC multi-threaded library for pricing financial derivatives.

## Python API

```python
>>> import polars as pl

>>> import rlib

>>> df = pl.read_csv('asset_data.csv')

>>> df.head()
shape: (5, 4)
┌────────────┬─────────────┬──────────┬──────────┐
│ strike     ┆ asset_price ┆ sigma    ┆ maturity │
│ ---        ┆ ---         ┆ ---      ┆ ---      │
│ f64        ┆ f64         ┆ f64      ┆ f64      │
╞════════════╪═════════════╪══════════╪══════════╡
│ 94.295197  ┆ 134.759569  ┆ 0.190168 ┆ 1.47368  │
│ 108.51493  ┆ 94.062597   ┆ 0.210614 ┆ 1.52059  │
│ 99.694597  ┆ 105.523219  ┆ 0.179769 ┆ 1.662726 │
│ 96.586621  ┆ 101.032656  ┆ 0.262634 ┆ 1.40694  │
│ 114.312273 ┆ 91.680799   ┆ 0.212748 ┆ 1.345785 │
└────────────┴─────────────┴──────────┴──────────┘

# run multi-threaded mc simulation with one million paths
>>> rlib.rust_par_mc_pricer(df, 0.01, 1_000_000)
shape: (10, 5)
┌────────────┬─────────────┬──────────┬──────────┬──────────────┐
│ strike     ┆ asset_price ┆ sigma    ┆ maturity ┆ option_price │
│ ---        ┆ ---         ┆ ---      ┆ ---      ┆ ---          │
│ f64        ┆ f64         ┆ f64      ┆ f64      ┆ f64          │
╞════════════╪═════════════╪══════════╪══════════╪══════════════╡
│ 94.295197  ┆ 134.759569  ┆ 0.190168 ┆ 1.47368  ┆ 42.503559    │
│ 108.51493  ┆ 94.062597   ┆ 0.210614 ┆ 1.52059  ┆ 5.192578     │
│ 99.694597  ┆ 105.523219  ┆ 0.179769 ┆ 1.662726 ┆ 13.609188    │
│ 96.586621  ┆ 101.032656  ┆ 0.262634 ┆ 1.40694  ┆ 15.241142    │
│ …          ┆ …           ┆ …        ┆ …        ┆ …            │
│ 94.194943  ┆ 82.47889    ┆ 0.192191 ┆ 1.834425 ┆ 4.908198     │
│ 95.2507    ┆ 88.184673   ┆ 0.191737 ┆ 1.72976  ┆ 6.665435     │
│ 101.523341 ┆ 87.219384   ┆ 0.23726  ┆ 1.34909  ┆ 5.053895     │
│ 98.659177  ┆ 124.518108  ┆ 0.203288 ┆ 1.4976   ┆ 29.581507    │
└────────────┴─────────────┴──────────┴──────────┴──────────────┘
```
## Compile from source

1. Install [Rust](https://www.rust-lang.org/) if you haven't already.
2. Active your Python env. The Python package will be installed here.
3. Install [maturin](https://maturin.rs/): `pip install maturin`.
4. In the root `rlib` directory, run `maturin develop --release`.

For local development, the `maturin develop` command is used to quickly build a package in debug mode by default and install it to your env. The `--release` flag compiles the Rust code with optimizations turned on.