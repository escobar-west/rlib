use polars::prelude::*;
use rand::{thread_rng, Rng};
use rand_distr::StandardNormal;

pub fn mc_pricer(df: DataFrame, rate: f64, n_sims: i32) -> PolarsResult<DataFrame> {
    let n_assets = df.height();
    let mut rng = thread_rng();
    let mut avg_price = 0.0.lit();
    let df = df.lazy();
    for _ in 0..n_sims {
        let z: Series = (&mut rng)
            .sample_iter::<f64, _>(StandardNormal)
            .take(n_assets)
            .collect();
        let rand_walk = col("sigma") * col("maturity").sqrt() * z.lit();
        let mean_drift: Expr = (rate.lit() - 0.5.lit() * col("sigma").pow(2)) * col("maturity");
        let paths = col("asset_price") * (mean_drift + rand_walk).exp();
        let payoff = ((-rate).lit() * col("maturity")).exp()
            * (paths - col("strike")).clip_min(AnyValue::Float64(0.0));
        avg_price = avg_price + payoff / (n_sims as f32).lit();
    }
    df.select(&[avg_price.alias("price")]).collect()
}
