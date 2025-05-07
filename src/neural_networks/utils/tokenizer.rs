use tokenizers::{AddedToken, Tokenizer};
use std::{error::Error, path::Path};

// https://huggingface.co/togethercomputer/GPT-NeoXT-Chat-Base-20B/tree/main
// "vocab_size": 50432 defined in https://huggingface.co/togethercomputer/GPT-NeoXT-Chat-Base-20B/blob/main/config.json
pub const TOKENIZER_GTP_NEOXT_PATH: &str = "src/neural_networks/tokenizers/gtp_neox_tokenizer.json";


pub fn tokenize_batch(text_batch: &Vec<String>, with_eos: bool) -> Result<(Vec<Vec<String>>, Vec<Vec<u32>>), Box<dyn Error + Send + Sync>> {
    let mut encoded_tokens_batch = Vec::new();
    let mut encoded_ids_batch = Vec::new();

    for text in text_batch {
        let (mut encoded_tokens, mut encoded_ids) = tokenize(text).unwrap();

        if with_eos {
            encoded_ids.push(0); // EOS token ID
            encoded_tokens.push("<|endoftext|>".to_string()); // Optional: track string tokens too
        }

        encoded_tokens_batch.push(encoded_tokens);
        encoded_ids_batch.push(encoded_ids);
    }

    Ok((encoded_tokens_batch, encoded_ids_batch))
}

pub fn tokenize(text: &str) -> Result<(Vec<String>, Vec<u32>), Box<dyn Error + Send + Sync>> {
    let path = Path::new(TOKENIZER_GTP_NEOXT_PATH).to_path_buf();
    // Load the pretrained tokenizer from the `tokenizer.json` file
    let tokenizer = Tokenizer::from_file(path).expect("Fehler bei Tokenizer");

    // Tokenize the text
    let encoding = tokenizer.encode(text, true)?;
    let encoded_tokens= encoding.get_tokens().to_vec();
    let encoded_ids = encoding.get_ids().to_vec();

    Ok((encoded_tokens, encoded_ids))
}

pub fn detokenize(ids: &Vec<u32>, skip_speical_tokens: bool) -> Result<String, Box<dyn Error + Send + Sync>> {
    let path = Path::new(TOKENIZER_GTP_NEOXT_PATH).to_path_buf();
    // Load the pretrained tokenizer from the `tokenizer.json` file
    let tokenizer = Tokenizer::from_file(path).expect("Fehler bei Tokenizer");

    // Tokenize the text
    let decoded_text = tokenizer.decode(ids, skip_speical_tokens)?;

    Ok(decoded_text)
}

pub fn create_custom_tokenizer() -> Result<Tokenizer, Box<dyn Error + Send + Sync>> {
    let path = Path::new(TOKENIZER_GTP_NEOXT_PATH).to_path_buf();
    // Load the pretrained tokenizer from the `tokenizer.json` file
    let mut tokenizer = Tokenizer::from_file(&path).expect("Fehler bei Tokenizer");

    // Add special tokens
    tokenizer.add_special_tokens(&[
        // begin of sentence
        AddedToken::from("<bos>", true),
        // separator
        AddedToken::from("<sep>", true),
        AddedToken::from("<eos>", true),
        AddedToken::from("<context>", true),
        AddedToken::from("</context>", true),
    ]);

    println!("Special tokens:");
    println!("<bos>: {:?}", tokenizer.token_to_id("<bos>"));
    println!("eos: {:?}", tokenizer.token_to_id("<eos>")); 
    println!("<|padding|>: {:?}", tokenizer.token_to_id("<|padding|>"));
    println!("sep: {:?}", tokenizer.token_to_id("<sep>"));

    tokenizer.save(&path, true)?;

    Ok(tokenizer)
}