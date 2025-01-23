use super::{monte_carlo_simulation::{simulate, MonteCarloParams}, stock_apis::fetch_and_save_historical_data, stock_ticker_symbols::NVIDIA};



pub fn referesh_stock_prices() {
    fetch_and_save_historical_data(NVIDIA);
    simulate_monte_carlo(NVIDIA);
}

pub fn simulate_monte_carlo(stock_price_name: &str) {
    let montecarlo_params = MonteCarloParams {
        num_simulations: 1000,
        year_to_predict: 2

    };
    simulate(montecarlo_params, stock_price_name);
}