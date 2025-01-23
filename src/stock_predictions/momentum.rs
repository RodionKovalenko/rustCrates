
use csv::ReaderBuilder;

pub fn calculate_moving_average(data: &[f64], window: usize) -> Vec<f64> {
    data.windows(window)
        .map(|window| window.iter().sum::<f64>() / window.len() as f64)
        .collect()
}


pub fn calculate_macd(data: &[f64], short_window: usize, long_window: usize, signal_window: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let short_ema = calculate_ema(data, short_window);
    let long_ema = calculate_ema(data, long_window);
    let macd_line: Vec<f64> = short_ema.iter().zip(long_ema.iter()).map(|(s, l)| s - l).collect();
    let signal_line = calculate_ema(&macd_line, signal_window);
    let histogram: Vec<f64> = macd_line.iter().zip(signal_line.iter()).map(|(m, s)| m - s).collect();
    (macd_line, signal_line, histogram)
}

pub fn calculate_ema(data: &[f64], period: usize) -> Vec<f64> {
    let mut ema = Vec::new();
    let multiplier = 2.0 / (period as f64 + 1.0);

    for (i, &price) in data.iter().enumerate() {
        if i == 0 {
            // Use the first price as the starting EMA
            ema.push(price);
        } else {
            // Calculate EMA: [Current Price - Previous EMA] * Multiplier + Previous EMA
            let previous_ema = ema[i - 1];
            let current_ema = (price - previous_ema) * multiplier + previous_ema;
            ema.push(current_ema);
        }
    }

    ema
}

pub fn calculate_rsi(data: &[f64], period: usize) -> Vec<f64> {
    let mut rsi = Vec::new();
    let mut gains = Vec::new();
    let mut losses = Vec::new();

    for i in 1..data.len() {
        let change = data[i] - data[i - 1];
        if change > 0.0 {
            gains.push(change);
            losses.push(0.0);
        } else {
            gains.push(0.0);
            losses.push(-change);
        }
    }

    let mut avg_gain = gains.iter().take(period).sum::<f64>() / period as f64;
    let mut avg_loss = losses.iter().take(period).sum::<f64>() / period as f64;

    rsi.push(100.0 - (100.0 / (1.0 + avg_gain / avg_loss)));

    for i in period..gains.len() {
        avg_gain = (avg_gain * (period as f64 - 1.0) + gains[i]) / period as f64;
        avg_loss = (avg_loss * (period as f64 - 1.0) + losses[i]) / period as f64;
        rsi.push(100.0 - (100.0 / (1.0 + avg_gain / avg_loss)));
    }

    rsi
}


fn backtest(data: &[f64], signals: &[bool]) -> f64 {
    let mut capital = 10000.0;
    let mut position = 0.0;

    for (price, &signal) in data.iter().zip(signals.iter()) {
        if signal {
            // Buy
            position += capital / price;
            capital = 0.0;
        } else if position > 0.0 {
            // Sell
            capital += position * price;
            position = 0.0;
        }
    }

    capital + position * data.last().unwrap_or(&0.0)
}



fn test() {
    // Read historical data
    let data = read_csv("data.csv");

    // Calculate indicators
    let ma = calculate_moving_average(&data, 50);
    let (macd, signal, histogram) = calculate_macd(&data, 12, 26, 9);
    let rsi = calculate_rsi(&data, 14);

    // // Generate signals
    // let signals = generate_signals(&ma, &macd, &rsi);

    // // Backtest
    // let final_capital = backtest(&data, &signals);
    // println!("Final capital after backtesting: {}", final_capital);

    // // Visualize
    // visualize(&data, &ma, &macd, &rsi, &signals);
}

fn read_csv(filename: &str) -> Vec<f64> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(filename)
        .expect("Failed to open file");

    reader
        .records()
        .map(|record| {
            record
                .unwrap()
                .get(4)
                .unwrap()
                .parse::<f64>()
                .expect("Invalid data")
        })
        .collect()
}

// Implement other functions here...
