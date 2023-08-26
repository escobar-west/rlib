mod rust_pricer;
use pyo3::prelude::*;
use pyo3_polars::{error::PyPolarsErr, PyDataFrame};

#[pyfunction]
fn mc_pricer(df: PyDataFrame) -> PyResult<PyDataFrame> {
    let rdf = rust_pricer::mc_pricer(df.into()).map_err(PyPolarsErr::from)?;
    Ok(PyDataFrame(rdf))
}

/// A Python module implemented in Rust.
#[pymodule]
fn rlib(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mc_pricer, m)?)?;
    Ok(())
}
