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
    df.select(&[avg_price.alias("option_price")]).collect()
}

pub fn mc_pricer_2(df: DataFrame, rate: f64, n_sims: i32) -> PolarsResult<DataFrame> {
    let n_assets = df.height();
    let mut rng = thread_rng();
    let sigma: Vec<f64> = df
        .column("sigma")?
        .f64()?
        .into_iter()
        .map(Option::unwrap)
        .collect();
    let strike: Vec<f64> = df
        .column("strike")?
        .f64()?
        .into_iter()
        .map(Option::unwrap)
        .collect();
    let asset_price: Vec<f64> = df
        .column("asset_price")?
        .f64()?
        .into_iter()
        .map(Option::unwrap)
        .collect();
    let sqrt_maturity: Vec<f64> = df
        .column("maturity")?
        .f64()?
        .into_iter()
        .map(|x| x.unwrap().sqrt())
        .collect();

    let mut rng_buffer: Vec<f64> = vec![0.0; n_assets];
    let mut option_prices: Vec<f64> = vec![0.0; n_assets];
    for _ in 0..n_sims {
        // fill in rng values for single sim
        for (buff, r) in rng_buffer
            .iter_mut()
            .zip((&mut rng).sample_iter::<f64, _>(StandardNormal))
        {
            *buff = r;
        }
        // update price with sim result (divide by n_sims comes later)
        for (price, r) in option_prices.iter_mut().zip(rng_buffer.iter()) {
            *price += r;
        }
    }
    for price in option_prices.iter_mut() {
        *price /= n_sims as f64;
    }
    let output: Series = option_prices.into_iter().collect();
    Ok(output.into_frame())
}
