use polars::prelude::*;
use rand::{thread_rng, Rng};
use rand_distr::StandardNormal;
use rayon::prelude::*;

pub fn mc_pricer(df: DataFrame, n_sims: i32) -> PolarsResult<DataFrame> {
    let mut rng = thread_rng();
    let mut accum_res = Vec::with_capacity(df.height()).resize(df.height(), 0f64);
    for _ in 0..n_sims {
        let z_vals: Series = (&mut rng)
            .sample_iter::<f64, _>(StandardNormal)
            .take(df.height())
            .collect();
        let rand_walk = col("sigma") * col("maturity") * z_vals.lit();
    }
    let rand_walk = (col("sigma") * col("maturity")).alias("rand_walk");
    df.lazy().with_column(rand_walk).collect()
}
