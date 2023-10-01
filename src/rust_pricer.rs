use polars::prelude::*;
use rand::{thread_rng, Rng};
use rand_distr::StandardNormal;
use rayon::prelude::*;
use statrs::distribution::{ContinuousCDF, Normal};

pub fn rust_ref_pricer(df: DataFrame, rate: f64) -> PolarsResult<DataFrame> {
    let rate = rate.lit();
    let var_factor = col("sigma") * col("maturity").sqrt();
    let d1 = (col("maturity") * (rate.clone() + 0.5.lit() * col("sigma") * col("sigma"))
        + (col("asset_price") / col("strike"))
            .map(|s| smap(s, f64::ln), std::default::Default::default()))
        / var_factor.clone();
    let d2 = d1.clone() - var_factor;
    let option_price = col("asset_price")
        * d1.map(
            |s| {
                let n = Normal::new(0.0, 1.0).unwrap();
                smap(s, |x| n.cdf(x))
            },
            std::default::Default::default(),
        )
        - col("strike")
            * ((-1).lit() * rate * col("maturity"))
                .map(|s| smap(s, f64::exp), std::default::Default::default())
            * d2.map(
                |s| {
                    let n = Normal::new(0.0, 1.0).unwrap();
                    smap(s, |x| n.cdf(x))
                },
                std::default::Default::default(),
            );
    let df = df
        .lazy()
        .with_column(option_price.alias("option_price"))
        .collect()?;
    Ok(df)
}
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
    // mean_drift = maturity * (rate - 0.5 * sigma**2)
    let mean_drift: Vec<f64> = std::iter::zip(sigma.iter(), maturity.iter())
        .map(|(sig, mat)| mat * (rate - 0.5 * sig * sig))
        .collect();
    // discount_rate = (-rate * maturity).exp()
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
        // start with z = standard normal rv for single path
        for (buff, z) in calc_buffer
            .iter_mut()
            .zip((&mut rng).sample_iter::<f64, _>(StandardNormal))
        {
            *buff = z;
        }
        // rand_walk = sigma * maturity.sqrt() * z;
        update_buffer(&mut calc_buffer, sqrt_maturity, |buff, m| *buff *= m);
        update_buffer(&mut calc_buffer, sigma, |buff, s| *buff *= s);
        // paths = asset_price * (mean_drift + rand_walk).exp();
        update_buffer(&mut calc_buffer, mean_drift, |buff, m| {
            *buff += m;
            *buff = (*buff).exp();
        });
        update_buffer(&mut calc_buffer, asset_price, |buff, a| *buff *= a);
        // payoff = discount_rate * max(paths - strike, 0);
        update_buffer(&mut calc_buffer, strike, |buff, s| {
            *buff -= s;
            *buff = (*buff).max(0.0);
        });
        update_buffer(&mut calc_buffer, discount_rate, |buff, d| *buff *= d);
        // update price with path result
        update_buffer(&mut price_buffer, &calc_buffer, |price, b| *price += b);
    }
    price_buffer
}

fn update_buffer<F>(buffer: &mut [f64], source: &[f64], mut func: F)
where
    F: FnMut(&mut f64, &f64),
{
    for (b, a) in buffer.iter_mut().zip(source.iter()) {
        func(b, a);
    }
}

fn smap<F>(s: Series, f: F) -> PolarsResult<Option<Series>>
where
    F: Fn(f64) -> f64,
{
    let out: Series = s.f64()?.into_iter().map(|x| f(x.unwrap())).collect();
    Ok(Some(out))
}
