use encoding_rs::WINDOWS_1252;

pub fn fix_encoding(s: &str) -> String {
    // Interpret the UTF-8 string as raw bytes, then decode as Windows-1252
    let (decoded, _, _) = WINDOWS_1252.decode(s.as_bytes());
    decoded.into_owned()
}