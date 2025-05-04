use serde::Deserialize;
use std::fs::File;
use std::io::Read;

#[derive(Debug, Deserialize)]
pub struct XQuADInstance {
    pub context: String,
    pub question: String,
    pub answers: Answer,
}

#[derive(Debug, Deserialize)]
pub struct Answer {
    pub text: Vec<String>, // Multiple answers may exist
}

#[derive(Debug, Deserialize)]
pub struct XQuADDataset {
    pub data: Vec<XQuADInstance>, // List of question-answer instances
}

pub fn load_data_xquad_de() -> Result<XQuADDataset, Box<dyn std::error::Error>> {
    let file_path = "datasets/xquad.de.json";
    let mut file = File::open(file_path)?;

    let mut content = String::new();
    file.read_to_string(&mut content)?;

    let dataset: XQuADDataset = serde_json::from_str(&content)?;

    println!("Total question-answer pairs: {}", dataset.data.len());

    Ok(dataset)
}
