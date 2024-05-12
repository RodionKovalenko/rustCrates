use std::process::Command;

pub fn make_prediction_daily() {
    println!("prediction are being made...");
    Command::new("cmd")
        .args(&["/C", "cd src/neural_networks && python feedforward-prediction.py && python lstm.py && python lstm-2.py && python prophet.py"])
        .output()
        .expect("failed to execute process");

    println!("predictions completed");
}