use polars::prelude::*;
use rand::{thread_rng, Rng};
use rand_distr::StandardNormal;
use rayon::prelude::*;

pub fn rust_mc_pricer(mut df: DataFrame, rate: f64, n_paths: i32) -> PolarsResult<DataFrame> {
    let n_assets = df.height();
    let sigma = get_as_vecf64(&df, "sigma")?;
    let strike = get_as_vecf64(&df, "strike")?;
    let asset_price = get_as_vecf64(&df, "asset_price")?;
    let maturity = get_as_vecf64(&df, "maturity")?;
    let sqrt_maturity: Vec<f64> = maturity.iter().map(|x| x.sqrt()).collect();
    // mean_drift = (rate - 0.5 * sigma**2) * maturity
    let mean_drift: Vec<f64> = std::iter::zip(sigma.iter(), maturity.iter())
        .map(|(sig, mat)| mat * (rate - 0.5 * sig * sig))
        .collect();
    // discount_rate = ((-rate).lit() * col("maturity")).exp()
    let discount_rate: Vec<f64> = maturity.iter().map(|mat| (-rate * mat).exp()).collect();
    let mut option_prices = run_mc_sim(
        n_assets,
        n_paths,
        &sigma,
        &strike,
        &asset_price,
        &sqrt_maturity,
        &mean_drift,
        &discount_rate,
    );
    // normalize by n_paths
    for price in option_prices.iter_mut() {
        *price /= n_paths as f64;
    }
    let mut output: Series = option_prices.into_iter().collect();
    output.rename("option_price");
    df.with_column(output)?;
    Ok(df)
}

pub fn rust_par_mc_pricer(mut df: DataFrame, rate: f64, n_paths: i32) -> PolarsResult<DataFrame> {
    let n_assets = df.height();
    let sigma = get_as_vecf64(&df, "sigma")?;
    let strike = get_as_vecf64(&df, "strike")?;
    let asset_price = get_as_vecf64(&df, "asset_price")?;
    let maturity = get_as_vecf64(&df, "maturity")?;
    let sqrt_maturity: Vec<f64> = maturity.iter().map(|x| x.sqrt()).collect();
    // mean_drift = (rate - 0.5 * sigma**2) * maturity
    let mean_drift: Vec<f64> = std::iter::zip(sigma.iter(), maturity.iter())
        .map(|(sig, mat)| mat * (rate - 0.5 * sig * sig))
        .collect();
    // discount_rate = ((-rate).lit() * col("maturity")).exp()
    let discount_rate: Vec<f64> = maturity.iter().map(|mat| (-rate * mat).exp()).collect();
    let n_cores = std::thread::available_parallelism()?.get() as i32;
    let paths_per_core = n_paths / n_cores;
    let mut option_prices = (0..n_cores)
        .into_par_iter()
        .map(|_| {
            // run one mc sim per core
            run_mc_sim(
                n_assets,
                paths_per_core,
                &sigma,
                &strike,
                &asset_price,
                &sqrt_maturity,
                &mean_drift,
                &discount_rate,
            )
        })
        .reduce(
            || vec![0.0; n_assets],
            |a: Vec<f64>, b: Vec<f64>| {
                std::iter::zip(a.iter(), b.iter())
                    .map(|(elem_a, elem_b)| elem_a + elem_b)
                    .collect()
            },
        );
    // normalize by number of paths that actually ran
    for price in option_prices.iter_mut() {
        *price /= (paths_per_core * n_cores) as f64;
    }
    let mut output: Series = option_prices.into_iter().collect();
    output.rename("option_price");
    df.with_column(output)?;
    Ok(df)
}

fn get_as_vecf64(df: &DataFrame, col: &str) -> PolarsResult<Vec<f64>> {
    Ok(df
        .column(col)?
        .f64()?
        .into_iter()
        .map(Option::unwrap)
        .collect())
}

#[allow(clippy::too_many_arguments)]
fn run_mc_sim(
    n_assets: usize,
    n_paths: i32,
    sigma: &[f64],
    strike: &[f64],
    asset_price: &[f64],
    sqrt_maturity: &[f64],
    mean_drift: &[f64],
    discount_rate: &[f64],
) -> Vec<f64> {
    let mut rng = thread_rng();
    let mut calc_buffer: Vec<f64> = vec![0.0; n_assets];
    let mut price_buffer: Vec<f64> = vec![0.0; n_assets];
    for _ in 0..n_paths {
        // fill in rng values for single path
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
        // update price with path result
        for (price, b) in price_buffer.iter_mut().zip(calc_buffer.iter()) {
            *price += b;
        }
    }
    price_buffer
}
