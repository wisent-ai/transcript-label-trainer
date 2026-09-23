use super::*;

/// `textwrap.wrap`: greedy fill over whitespace- and hyphen-separated chunks,
/// dropping the whitespace a line breaks on.
pub(crate) fn wrap(text: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    let mut chunks: Vec<String> = Vec::new();
    for (index, word) in text.split_whitespace().enumerate() {
        if index > 0 {
            chunks.push(" ".to_string());
        }
        chunks.extend(hyphen_chunks(word));
    }

    let mut lines: Vec<String> = Vec::new();
    let mut position = 0;
    while position < chunks.len() {
        if !lines.is_empty() && chunks[position] == " " {
            position += 1;
            continue;
        }
        let mut current: Vec<String> = Vec::new();
        let mut length = 0;
        while position < chunks.len() {
            let chunk_length = chunks[position].chars().count();
            if length + chunk_length > width {
                break;
            }
            length += chunk_length;
            current.push(chunks[position].clone());
            position += 1;
        }
        // One chunk wider than the whole line: cut it and keep the remainder.
        if position < chunks.len() && chunks[position].chars().count() > width {
            let room = width.saturating_sub(length).max(1);
            let chunk = chunks[position].clone();
            current.push(chunk.chars().take(room).collect());
            chunks[position] = chunk.chars().skip(room).collect();
        }
        while current.last().is_some_and(|chunk| chunk == " ") {
            current.pop();
        }
        if current.is_empty() {
            position += 1;
            continue;
        }
        lines.push(current.concat());
    }
    lines
}

/// A word split at the hyphens `textwrap` is willing to break on, the hyphen
/// staying with the chunk before it.
pub(crate) fn hyphen_chunks(word: &str) -> Vec<String> {
    let characters: Vec<char> = word.chars().collect();
    let mut chunks: Vec<String> = Vec::new();
    let mut start = 0;
    for index in 0..characters.len() {
        if characters[index] != '-' || !breakable_hyphen(&characters, index) {
            continue;
        }
        chunks.push(characters[start..=index].iter().collect());
        start = index + 1;
    }
    if start < characters.len() {
        chunks.push(characters[start..].iter().collect());
    }
    if chunks.is_empty() {
        chunks.push(word.to_string());
    }
    chunks
}

pub(crate) fn breakable_hyphen(characters: &[char], index: usize) -> bool {
    let letter = |at: usize| {
        characters
            .get(at)
            .is_some_and(|character| character.is_alphabetic() || *character == '_')
    };
    let behind = (index >= 2 && letter(index - 1) && letter(index - 2))
        || (index >= 3
            && letter(index - 1)
            && characters.get(index - 2) == Some(&'-')
            && letter(index - 3));
    if !behind || !letter(index + 1) {
        return false;
    }
    letter(index + 2) || (characters.get(index + 2) == Some(&'-') && letter(index + 3))
}

pub(crate) fn top_error(message: &str) -> i32 {
    let specs = build_specs();
    let (optionals, positionals) = top_usage_parts(&specs);
    eprint!("{}", format_usage(PROG, &optionals, &positionals));
    eprintln!("{PROG}: error: {message}");
    2
}

pub(crate) fn sub_error(spec: &Spec, message: &str) -> i32 {
    let prog = format!("{PROG} {}", spec.name);
    let (optionals, positionals) = sub_usage_parts(spec);
    eprint!("{}", format_usage(&prog, &optionals, &positionals));
    eprintln!("{prog}: error: {message}");
    2
}
