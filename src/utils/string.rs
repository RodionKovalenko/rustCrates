use encoding_rs::WINDOWS_1252;

pub fn maybe_fix_encoding(s: &str) -> String {
    // If it already decodes correctly as UTF-8, leave it alone.
    if let Ok(_) = std::str::from_utf8(s.as_bytes()) {
        return s.to_string();
    }

    // Try interpreting the *bytes* as Windows-1252, assuming they were wrongly decoded
    let (decoded, _, _) = WINDOWS_1252.decode(s.as_bytes());
    decoded.into_owned()
}