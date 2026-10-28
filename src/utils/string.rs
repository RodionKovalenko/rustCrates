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

/// Turns free-form user input into the order the question-answering model was trained on
/// (`qa_input`: the text, then the question on its own line):
/// * JSON escapes pasted from a raw JSON file (`f\u00fcr`, `\n`, `\"`) become the real characters;
/// * the sentences ending with `?` are moved behind the rest of the text, so the question may be typed
///   before, inside or after the text. Input without a question, or with nothing but questions, keeps its order.
pub fn prepare_prompt(input: &str) -> String {
    let text = decode_json_escapes(input.trim());
    let sentences = split_sentences(&text);
    let (questions, rest): (Vec<&str>, Vec<&str>) = sentences.iter().partition(|s| s.ends_with('?'));
    if questions.is_empty() || rest.is_empty() {
        return text;
    }
    format!("{}\n{}", rest.join(" "), questions.join(" "))
}

/// Decodes `\uXXXX` (including surrogate pairs), `\n`, `\t`, `\"`, `\/` and `\\`; any other backslash stays.
pub fn decode_json_escapes(s: &str) -> String {
    if !s.contains('\\') {
        return s.to_string();
    }
    let chars: Vec<char> = s.chars().collect();
    let hex4 = |at: usize| -> Option<u32> {
        if at + 4 > chars.len() {
            return None;
        }
        let digits: String = chars[at..at + 4].iter().collect();
        u32::from_str_radix(&digits, 16).ok()
    };
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '\\' && i + 1 < chars.len() {
            match chars[i + 1] {
                'u' => {
                    if let Some(code) = hex4(i + 2) {
                        // A high surrogate needs the low one that follows (`\ud83d\ude00`).
                        if (0xD800..0xDC00).contains(&code) && i + 7 < chars.len() && chars[i + 6] == '\\' && chars[i + 7] == 'u' {
                            if let Some(low) = hex4(i + 8).filter(|low| (0xDC00..0xE000).contains(low)) {
                                if let Some(c) = char::from_u32(0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)) {
                                    out.push(c);
                                    i += 12;
                                    continue;
                                }
                            }
                        }
                        if let Some(c) = char::from_u32(code) {
                            out.push(c);
                            i += 6;
                            continue;
                        }
                    }
                }
                'n' => {
                    out.push('\n');
                    i += 2;
                    continue;
                }
                't' => {
                    out.push('\t');
                    i += 2;
                    continue;
                }
                '"' | '\\' | '/' => {
                    out.push(chars[i + 1]);
                    i += 2;
                    continue;
                }
                _ => {}
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

/// Abbreviations (lower case, without the dot) whose dot does not end a sentence.
const ABBREVIATIONS: [&str; 30] = [
    "nr", "dr", "prof", "ca", "bzw", "usw", "etc", "vgl", "ggf", "evtl", "inkl", "z", "st", "hr", "fr", "jh", "jhd", "mio", "mrd",
    "tsd", "bspw", "sog", "u", "o", "mr", "mrs", "ms", "vs", "co", "abs",
];

/// Splits text into trimmed sentences. A sentence ends at a line break, at `?` / `!` before whitespace, and at
/// `.` before whitespace unless the word before the dot is a single letter ("z. B.", "d. h.") or a common
/// abbreviation ("Nr.", "Dr."), so a question containing an abbreviation stays one sentence.
fn split_sentences(text: &str) -> Vec<&str> {
    let mut sentences = Vec::new();
    let mut start = 0;
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    for (k, &(i, c)) in chars.iter().enumerate() {
        let next_is_space = chars.get(k + 1).map_or(true, |&(_, n)| n.is_whitespace());
        let end = match c {
            '\n' => true,
            '?' | '!' => next_is_space,
            '.' => next_is_space && {
                let word: String = text[start..i].chars().rev().take_while(|ch| ch.is_alphabetic()).collect::<Vec<_>>().into_iter().rev().collect();
                // A single letter ("z. B.", "d. h.") or a common abbreviation ("Nr.", "Dr.") does not end a sentence.
                !(word.chars().count() == 1 || ABBREVIATIONS.contains(&word.to_lowercase().as_str()))
            },
            _ => false,
        };
        if end {
            let end_byte = i + c.len_utf8();
            let sentence = text[start..end_byte].trim();
            if !sentence.is_empty() {
                sentences.push(sentence);
            }
            start = end_byte;
        }
    }
    let tail = text[start..].trim();
    if !tail.is_empty() {
        sentences.push(tail);
    }
    sentences
}

#[cfg(test)]
mod tests {
    use super::{decode_json_escapes, prepare_prompt};

    const TEXT: &str = "Die Panthers gaben nur 308 Punkte ab. Sie belegten den sechsten Platz.";

    #[test]
    fn question_after_the_text_goes_on_its_own_line() {
        assert_eq!(
            prepare_prompt(&format!("{TEXT} Wie viele Punkte gaben die Panthers ab?")),
            format!("{TEXT}\nWie viele Punkte gaben die Panthers ab?")
        );
    }

    #[test]
    fn question_before_or_inside_the_text_is_moved_to_the_end() {
        let expected = format!("{TEXT}\nWie viele Punkte gaben die Panthers ab?");
        assert_eq!(prepare_prompt(&format!("Wie viele Punkte gaben die Panthers ab? {TEXT}")), expected);
        assert_eq!(prepare_prompt("Die Panthers gaben nur 308 Punkte ab. Wie viele Punkte gaben die Panthers ab?\nSie belegten den sechsten Platz."), expected);
    }

    #[test]
    fn input_in_the_training_format_is_unchanged() {
        let input = format!("{TEXT}\nWie viele Punkte gaben die Panthers ab?");
        assert_eq!(prepare_prompt(&input), input);
    }

    #[test]
    fn input_without_question_or_only_a_question_keeps_its_order() {
        assert_eq!(prepare_prompt(&format!("Nenne die Punkte. {TEXT}")), format!("Nenne die Punkte. {TEXT}"));
        assert_eq!(prepare_prompt("  Wer gewann den Super Bowl?  "), "Wer gewann den Super Bowl?");
    }

    #[test]
    fn abbreviations_do_not_split_a_question() {
        assert_eq!(
            prepare_prompt("Wann wurde z. B. Nr. 5 gebaut? Das Haus Nr. 5 wurde 1901 gebaut."),
            "Das Haus Nr. 5 wurde 1901 gebaut.\nWann wurde z. B. Nr. 5 gebaut?"
        );
    }

    #[test]
    fn json_escapes_from_a_pasted_json_file_are_decoded() {
        assert_eq!(decode_json_escapes(r#"f\u00fcr regul\u00e4re \"Spielzeit\"\nNeu"#), "für reguläre \"Spielzeit\"\nNeu");
        assert_eq!(decode_json_escapes(r"\ud83d\ude00 C:\Pfad"), "😀 C:\\Pfad");
        assert_eq!(prepare_prompt(r"Er kam f\u00fcr die Saison. Wof\u00fcr kam er?"), "Er kam für die Saison.\nWofür kam er?");
    }
}
