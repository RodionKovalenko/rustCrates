use serde::Deserialize;
use std::fs::File;
use std::io::Read;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct XQuADDataset {
    pub data: Vec<Article>,
}

#[derive(Debug, Deserialize)]
pub struct Article {
    pub paragraphs: Vec<Paragraph>,
}

#[derive(Debug, Deserialize)]
pub struct Paragraph {
    pub context: String,
    pub qas: Vec<QuestionAnswer>,
}

#[derive(Debug, Deserialize)]
pub struct QuestionAnswer {
    pub question: String,
    pub answers: Vec<Answer>,
}

#[derive(Debug, Deserialize)]
pub struct Answer {
    pub text: String,
}

pub fn load_data_xquad_de() -> Result<XQuADDataset, Box<dyn std::error::Error>> {
    let file_path = "datasets/xquad.de.json";
    let mut file = File::open(file_path)?;

    if !Path::new(file_path).exists() {
        return Err(format!("Dataset file not found at path: {}", file_path).into());
    }

    let mut content = String::new();
    file.read_to_string(&mut content)?;

    let dataset: XQuADDataset = serde_json::from_str(&content)?;

    println!("Valid question-answer pairs: {}", dataset.data.len());

    Ok(dataset)
}
