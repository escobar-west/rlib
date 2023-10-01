mod rust_pricer;
use pyo3::prelude::*;
use pyo3_polars::{error::PyPolarsErr, PyDataFrame};

#[pyfunction]
fn rust_mc_pricer(df: PyDataFrame, rate: f64, n_paths: i32) -> PyResult<PyDataFrame> {
    let rdf = rust_pricer::rust_mc_pricer(df.into(), rate, n_paths).map_err(PyPolarsErr::from)?;
    Ok(PyDataFrame(rdf))
}

#[pyfunction]
fn rust_par_mc_pricer(df: PyDataFrame, rate: f64, n_paths: i32) -> PyResult<PyDataFrame> {
    let rdf =
        rust_pricer::rust_par_mc_pricer(df.into(), rate, n_paths).map_err(PyPolarsErr::from)?;
    Ok(PyDataFrame(rdf))
}

/// A Python module implemented in Rust.
#[pymodule]
fn rlib(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rust_mc_pricer, m)?)?;
    m.add_function(wrap_pyfunction!(rust_par_mc_pricer, m)?)?;
    Ok(())
}
