use polars::prelude::*;
use rand::{thread_rng, Rng};
use rand_distr::StandardNormal;

pub fn rust_mc_pricer(mut df: DataFrame, rate: f64, n_sims: i32) -> PolarsResult<DataFrame> {
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
    let maturity: Vec<f64> = df
        .column("maturity")?
        .f64()?
        .into_iter()
        .map(Option::unwrap)
        .collect();
    let sqrt_maturity: Vec<f64> = maturity.iter().map(|x| x.sqrt()).collect();
    // mean_drift = (rate - 0.5 * sigma**2) * maturity
    let mean_drift: Vec<f64> = std::iter::zip(sigma.iter(), maturity.iter())
        .map(|(sig, mat)| mat * (rate - 0.5 * sig * sig))
        .collect();
    // discount_rate = ((-rate).lit() * col("maturity")).exp()
    let discount_rate: Vec<f64> = maturity.iter().map(|mat| (-rate * mat).exp()).collect();
    // calc_buffer and option_prices are buffers to store calculations through sims
    let mut calc_buffer: Vec<f64> = vec![0.0; n_assets];
    let mut option_prices: Vec<f64> = vec![0.0; n_assets];
    for _ in 0..n_sims {
        // fill in rng values for single sim
        for (buff, r) in calc_buffer
            .iter_mut()
            .zip((&mut rng).sample_iter::<f64, _>(StandardNormal))
        {
            *buff = r;
        }
        // let rand_walk = col("sigma") * col("maturity").sqrt() * z.lit();
        for (buff, m) in calc_buffer.iter_mut().zip(sqrt_maturity.iter()) {
            *buff *= m;
        }
        for (buff, s) in calc_buffer.iter_mut().zip(sigma.iter()) {
            *buff *= s;
        }
        // let paths = col("asset_price") * (mean_drift + rand_walk).exp();
        for (buff, m) in calc_buffer.iter_mut().zip(mean_drift.iter()) {
            *buff += m;
            *buff = (*buff).exp();
        }
        for (buff, a) in calc_buffer.iter_mut().zip(asset_price.iter()) {
            *buff *= a;
        }
        // payoff = discount_rate * (paths - col("strike")).clip_min(AnyValue::Float64(0.0));
        for (buff, s) in calc_buffer.iter_mut().zip(strike.iter()) {
            *buff -= s;
            *buff = (*buff).max(0.0);
        }
        for (buff, d) in calc_buffer.iter_mut().zip(discount_rate.iter()) {
            *buff *= d;
        }
        // update price with sim result
        for (price, b) in option_prices.iter_mut().zip(calc_buffer.iter()) {
            *price += b;
        }
    }
    // normalize by n_sims
    for price in option_prices.iter_mut() {
        *price /= n_sims as f64;
    }
    let mut output: Series = option_prices.into_iter().collect();
    output.rename("option_price");
    df.with_column(output)?;
    Ok(df)
}
