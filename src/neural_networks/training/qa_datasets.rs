//! Question-answering data in the SQuAD format (`{"data": [{"title", "paragraphs": [{"context", "qas": [{"question",
//! "answers": [{"text"}]}]}]}]}`): XQuAD (de) plus every `.json` file in `datasets/qa/`, e.g. GermanQuAD or MLQA split
//! into parts small enough for GitHub (`split_squad_file`).

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use crate::neural_networks::network_components::input::QaRecord;
use crate::neural_networks::training::xquad_structs::XQuADDataset;

/// Folder with additional SQuAD-format files; all of them are trained on together with XQuAD (de).
pub const QA_DATASET_DIR: &str = "datasets/qa";

/// Default size of one part written by `split_squad_file` (GitHub warns from 50 MB and refuses 100 MB).
pub const DEFAULT_PART_MB: usize = 10;

/// The `.json` files of `dir`, sorted by name (missing folder: none).
pub fn squad_files(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = fs::read_dir(dir)
        .map(|entries| entries.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().is_some_and(|e| e == "json")).collect())
        .unwrap_or_default();
    files.sort();
    files
}

/// The questions of one SQuAD-format file. Questions without an answer (SQuAD 2.0 "impossible" questions) are skipped.
pub fn squad_records(path: &Path) -> Result<Vec<QaRecord>, Box<dyn std::error::Error>> {
    let dataset: XQuADDataset = serde_json::from_str(&fs::read_to_string(path)?).map_err(|e| format!("{}: {}", path.display(), e))?;
    let mut records = Vec::new();
    for article in dataset.data {
        for paragraph in article.paragraphs {
            for qa in paragraph.qas {
                if let Some(answer) = qa.answers.first() {
                    records.push(QaRecord { context: paragraph.context.clone(), question: qa.question, answer: answer.text.clone() });
                }
            }
        }
    }
    Ok(records)
}

/// The questions of every file in `dir`, in file order.
pub fn qa_records_from(dir: &Path) -> Result<Vec<QaRecord>, Box<dyn std::error::Error>> {
    let mut records = Vec::new();
    for file in squad_files(dir) {
        let file_records = squad_records(&file)?;
        println!("{}: {} questions", file.display(), file_records.len());
        records.extend(file_records);
    }
    Ok(records)
}

/// Splits a SQuAD-format file into parts of at most `max_bytes` (as far as a single paragraph allows), written to
/// `out_dir` as `<name>_001.json`, `<name>_002.json`, ... Every part is a complete SQuAD file; an article larger than
/// a part is continued in the next part under the same title. Returns the written files.
pub fn split_squad_file(input: &Path, out_dir: &Path, name: &str, max_bytes: usize) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let root: Value = serde_json::from_str(&fs::read_to_string(input)?)?;
    let version = root.get("version").cloned().unwrap_or(Value::Null);
    let articles = root.get("data").and_then(Value::as_array).ok_or("no \"data\" array: not a SQuAD file")?;

    // Paragraph-sized units in file order: (title, paragraph).
    let mut units: Vec<(Value, Value)> = Vec::new();
    for article in articles {
        let title = article.get("title").cloned().unwrap_or(Value::Null);
        for paragraph in article.get("paragraphs").and_then(Value::as_array).into_iter().flatten() {
            units.push((title.clone(), paragraph.clone()));
        }
    }

    fs::create_dir_all(out_dir)?;
    let mut written = Vec::new();
    let mut part: Vec<(Value, Value)> = Vec::new();
    let mut part_bytes = 0usize;
    let flush = |part: &mut Vec<(Value, Value)>, written: &mut Vec<PathBuf>| -> Result<(), Box<dyn std::error::Error>> {
        if part.is_empty() {
            return Ok(());
        }
        // Consecutive paragraphs of the same article stay one article.
        let mut data: Vec<Value> = Vec::new();
        for (title, paragraph) in part.drain(..) {
            match data.last_mut() {
                Some(last) if last["title"] == title => last["paragraphs"].as_array_mut().expect("paragraphs").push(paragraph),
                _ => data.push(json!({ "title": title, "paragraphs": [paragraph] })),
            }
        }
        let path = out_dir.join(format!("{}_{:03}.json", name, written.len() + 1));
        fs::write(&path, serde_json::to_string(&json!({ "version": version, "data": data }))?)?;
        written.push(path);
        Ok(())
    };
    for (title, paragraph) in units {
        let bytes = serde_json::to_string(&paragraph)?.len() + 64;
        if part_bytes + bytes > max_bytes && !part.is_empty() {
            flush(&mut part, &mut written)?;
            part_bytes = 0;
        }
        part_bytes += bytes;
        part.push((title, paragraph));
    }
    flush(&mut part, &mut written)?;
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn squad(articles: usize, paragraphs: usize, questions: usize) -> Value {
        let data: Vec<Value> = (0..articles)
            .map(|a| {
                let paragraphs: Vec<Value> = (0..paragraphs)
                    .map(|p| {
                        let qas: Vec<Value> = (0..questions)
                            .map(|q| json!({ "id": format!("{a}-{p}-{q}"), "question": format!("Frage {a} {p} {q}?"), "answers": [{ "text": format!("Antwort {a} {p} {q}"), "answer_start": 0 }] }))
                            .collect();
                        json!({ "context": format!("Artikel {a}, Absatz {p}. ").repeat(20), "qas": qas })
                    })
                    .collect();
                json!({ "title": format!("Artikel {a}"), "paragraphs": paragraphs })
            })
            .collect();
        json!({ "version": "1.0", "data": data })
    }

    #[test]
    fn split_keeps_every_question_in_order_and_respects_the_part_size() {
        let dir = std::env::temp_dir().join(format!("qa_split_test_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let input = dir.join("big.json");
        fs::write(&input, serde_json::to_string(&squad(7, 9, 3)).unwrap()).unwrap();

        let out = dir.join("parts");
        let max_bytes = 8_000;
        let parts = split_squad_file(&input, &out, "big", max_bytes).unwrap();
        assert!(parts.len() > 3, "the file must be split into several parts");
        for part in &parts {
            assert!(fs::metadata(part).unwrap().len() as usize <= max_bytes + 1_000, "{} too large", part.display());
        }
        assert_eq!(parts[0].file_name().unwrap(), "big_001.json");

        // The parts together hold exactly the questions of the original, in the same order.
        let original = squad_records(&input).unwrap();
        let split = qa_records_from(&out).unwrap();
        assert_eq!(split.len(), 7 * 9 * 3);
        assert_eq!(original.iter().map(|r| &r.question).collect::<Vec<_>>(), split.iter().map(|r| &r.question).collect::<Vec<_>>());
        assert_eq!(original.iter().map(|r| &r.answer).collect::<Vec<_>>(), split.iter().map(|r| &r.answer).collect::<Vec<_>>());

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn questions_without_answer_are_skipped_and_a_missing_folder_is_empty() {
        let dir = std::env::temp_dir().join(format!("qa_records_test_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let mut data = squad(1, 1, 2);
        data["data"][0]["paragraphs"][0]["qas"][1]["answers"] = json!([]);
        fs::write(dir.join("a.json"), serde_json::to_string(&data).unwrap()).unwrap();
        fs::write(dir.join("notes.txt"), "not a dataset").unwrap();
        assert_eq!(qa_records_from(&dir).unwrap().len(), 1);
        assert!(qa_records_from(&dir.join("missing")).unwrap().is_empty());
        fs::remove_dir_all(&dir).unwrap();
    }
}
