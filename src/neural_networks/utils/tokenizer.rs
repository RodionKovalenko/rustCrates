use tokenizers::{AddedToken, Tokenizer};
use std::{error::Error, path::Path};
use std::sync::OnceLock;

// https://huggingface.co/togethercomputer/GPT-NeoXT-Chat-Base-20B/tree/main
// "vocab_size": 50432 defined in https://huggingface.co/togethercomputer/GPT-NeoXT-Chat-Base-20B/blob/main/config.json
pub const TOKENIZER_GTP_NEOXT_PATH: &str = "src/neural_networks/tokenizers/gtp_neox_tokenizer.json";

/// Tokenizer file chosen before the first use (`use_tokenizer`); the GPT-NeoX tokenizer otherwise.
static TOKENIZER_CHOICE: OnceLock<String> = OnceLock::new();
static TOKENIZER_CELL: OnceLock<(String, Tokenizer)> = OnceLock::new();

/// The tokenizer, loaded once (cached to avoid loading from disk on every call).
fn tokenizer() -> &'static Tokenizer {
    &TOKENIZER_CELL
        .get_or_init(|| {
            let path = TOKENIZER_CHOICE.get().cloned().unwrap_or_else(|| TOKENIZER_GTP_NEOXT_PATH.to_string());
            let tokenizer = Tokenizer::from_file(Path::new(&path)).unwrap_or_else(|e| panic!("Failed to load tokenizer {path}: {e}"));
            (path, tokenizer)
        })
        .1
}

/// Selects the tokenizer file (e.g. the one a model was trained with). Must come before the first tokenization;
/// choosing the file already in use is fine, another one is an error.
pub fn use_tokenizer(path: &str) -> Result<(), String> {
    if let Some((current, _)) = TOKENIZER_CELL.get() {
        return if current == path { Ok(()) } else { Err(format!("tokenizer {current} is already in use, cannot switch to {path}")) };
    }
    match TOKENIZER_CHOICE.set(path.to_string()) {
        Ok(()) => Ok(()),
        Err(_) if TOKENIZER_CHOICE.get().is_some_and(|p| p == path) => Ok(()),
        Err(_) => Err(format!("another tokenizer than {path} was already chosen")),
    }
}


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
    // Use the cached tokenizer
    let encoding = tokenizer().encode(text, true)?;
    let encoded_tokens= encoding.get_tokens().to_vec();
    let encoded_ids = encoding.get_ids().to_vec();

    Ok((encoded_tokens, encoded_ids))
}

/// Token ids of `text` (no special tokens added) with the byte range of `text` each token covers.
pub fn encode_with_offsets(text: &str) -> Result<(Vec<u32>, Vec<(usize, usize)>), Box<dyn Error + Send + Sync>> {
    let encoding = tokenizer().encode(text, false)?;
    Ok((encoding.get_ids().to_vec(), encoding.get_offsets().to_vec()))
}

/// Id of a token of the vocabulary (e.g. `"<eos>"`), `None` if it does not exist.
pub fn token_id(token: &str) -> Option<u32> {
    tokenizer().token_to_id(token)
}

pub fn detokenize(ids: &Vec<u32>, skip_speical_tokens: bool) -> Result<String, Box<dyn Error + Send + Sync>> {
    // Use the cached tokenizer
    let decoded_text = tokenizer().decode(ids, skip_speical_tokens)?;
    Ok(decoded_text)
}

pub fn create_custom_tokenizer() -> Result<Tokenizer, Box<dyn Error + Send + Sync>> {
    let path = Path::new(TOKENIZER_GTP_NEOXT_PATH).to_path_buf();
    // Load the pretrained tokenizer from the `tokenizer.json` file
    let mut tokenizer = Tokenizer::from_file(&path).expect("Fehler bei Tokenizer");

    // Add special tokens
    let _ = tokenizer.add_special_tokens([
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