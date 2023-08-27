use polars::prelude::*;
use rand::{thread_rng, Rng};
use rand_distr::StandardNormal;

pub fn mc_pricer(df: DataFrame, n_sims: i32) -> PolarsResult<DataFrame> {
    let mut rng = thread_rng();
    for _ in 0..n_sims {
        let _z: Vec<f64> = (&mut rng)
            .sample_iter(StandardNormal)
            .take(df.height())
            .collect();
    }
    Ok(df)
}
