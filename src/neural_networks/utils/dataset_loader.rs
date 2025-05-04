use std::{fs::File, io::Write, path::Path};

#[tokio::main]
pub async fn load_dataset_xquad() -> Result<(), Box<dyn std::error::Error>> {
    // URLs for the XQuAD dataset in different languages
    let urls = vec![
        // ("https://raw.githubusercontent.com/deepmind/xquad/master/xquad.en.json", "xquad.en.json"),
        // ("https://raw.githubusercontent.com/deepmind/xquad/master/xquad.de.json", "xquad.de.json"),
        // ("https://raw.githubusercontent.com/deepmind/xquad/master/xquad.ru.json", "xquad.ru.json"),
        ("https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json", "xquad_en_train.json"),
        ("https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json", "xquad_en_validation.json"),
    ];

    let dataset_dir = Path::new("datasets");
    if !dataset_dir.exists() {
        std::fs::create_dir_all(dataset_dir)?;
    }

    for (url, file_name) in urls {
        // Define the path where the file will be saved
        let path = dataset_dir.join(file_name);

        // Send HTTP GET request to download the file
        let response = reqwest::get(url).await?;

        // Check if the request was successful
        if response.status().is_success() {
            // Create the file where we'll save the data inside the 'dataset' directory
            let mut file = File::create(path)?;

            // Write the content to the file
            let content = response.text().await?;
            file.write_all(content.as_bytes())?;

            println!("Downloaded the JSON data to 'dataset/{}'", file_name);
        } else {
            eprintln!("Failed to download file from '{}'. Status: {}", url, response.status());
        }
    }

    Ok(())
}
