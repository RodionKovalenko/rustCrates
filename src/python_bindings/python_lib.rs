use crate::neural_networks::training::train_transformer::train_transformer_from_dataset;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::wrap_pyfunction;

#[pyfunction]
fn train(num_epoch: usize, num_record: usize, batch_size: usize) -> PyResult<String> {
    train_transformer_from_dataset(num_epoch, num_record, batch_size);
    Ok("Training complete".to_string())
}

#[pymodule]
fn neural_networks(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(train, _py)?)?;
    Ok(())
}
