import pyperf
import polars as pl
import rlib
import pathlib

filepath = pathlib.Path(__file__).parent.resolve().joinpath("asset_data.csv")
pldf = pl.read_csv(filepath)
pddf = pldf.to_pandas()
RATE = 0.01
N_PATHS = 100_000


def bench_py_pandas_pricer():
    _ = rlib.py_pandas_pricer(pddf, RATE)

def bench_py_ref_pricer():
    _ = rlib.py_ref_pricer(pldf, RATE)

def bench_rust_ref_pricer():
    _ = rlib.rust_ref_pricer(pldf, RATE)

def bench_py_mc_pricer():
    _ = rlib.py_mc_pricer(pldf, RATE, N_PATHS)

def bench_rust_mc_pricer():
    _ = rlib.rust_mc_pricer(pldf, RATE, N_PATHS)

def bench_rust_par_mc_pricer():
    _ = rlib.rust_par_mc_pricer(pldf, RATE, N_PATHS)

runner = pyperf.Runner(values=1, processes=1)
#runner.bench_func("bench_py_pandas_pricer", bench_py_pandas_pricer)
#runner.bench_func("bench_py_ref_pricer", bench_py_ref_pricer)
#runner.bench_func("bench_rust_ref_pricer", bench_rust_ref_pricer)
runner.bench_func('bench_py_mc_pricer', bench_py_mc_pricer)
runner.bench_func('bench_rust_mc_pricer', bench_rust_mc_pricer)
runner.bench_func('bench_rust_par_mc_pricer', bench_rust_par_mc_pricer)