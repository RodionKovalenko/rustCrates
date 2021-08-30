use std::process::Command;

pub fn make_prediction_daily() {
    println!("prediction are being made...");
    let command1 = Command::new("cmd")
        .args(&["/C", "cd src/neural_networks && python feedforward-prediction.py"])
        .output()
        .expect("failed to execute process");

    let command2 = Command::new("cmd")
        .args(&["/C", "cd src/neural_networks && python lstm.py"])
        .output()
        .expect("failed to execute process");

    let command2 = Command::new("cmd")
        .args(&["/C", "cd src/neural_networks && python lstm-2.py"])
        .output()
        .expect("failed to execute process");

    let command4 = Command::new("cmd")
        .args(&["/C", "cd src/neural_networks && python prophet.py"])
        .output()
        .expect("failed to execute process");

    println!("predictions completed");
}