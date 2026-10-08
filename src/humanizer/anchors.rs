//! The facts a rewrite must carry over verbatim: URLs, e-mail addresses and
//! numbers. A generated source that drops one of the target's anchors would
//! teach the student to invent it, so such a pair is refused.
//!
//! This is the scan the Python preparation ran as the regular expression
//! `https?://\S+|\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b|\b\d[\d.,:/%-]*\b`, tried
//! alternative by alternative at each position from left to right, without
//! overlapping matches. The crate carries no regex engine, so it is written
//! out by hand.

use std::collections::HashSet;

/// Shortest top-level domain an e-mail anchor may end in.
const MIN_TLD_LETTERS: usize = 2;

fn word(character: Option<&char>) -> bool {
    character.is_some_and(|value| value.is_alphanumeric() || *value == '_')
}

fn boundary(chars: &[char], index: usize) -> bool {
    let before = index
        .checked_sub(1)
        .and_then(|previous| chars.get(previous));
    word(before) != word(chars.get(index))
}

fn run(chars: &[char], start: usize, member: impl Fn(char) -> bool) -> usize {
    chars[start..]
        .iter()
        .position(|value| !member(*value))
        .map_or(chars.len(), |offset| start + offset)
}

fn url_end(chars: &[char], start: usize) -> Option<usize> {
    let rest: String = chars[start..].iter().take("https://".len()).collect();
    let scheme = ["https://", "http://"]
        .into_iter()
        .find(|scheme| rest.starts_with(scheme))?;
    let body = start + scheme.len();
    let end = run(chars, body, |value| !value.is_whitespace());
    (end > body).then_some(end)
}

fn email_end(chars: &[char], start: usize) -> Option<usize> {
    if !boundary(chars, start) {
        return None;
    }
    let local = run(chars, start, |value| {
        value.is_alphanumeric() || "_.+-".contains(value)
    });
    if local == start || chars.get(local) != Some(&'@') {
        return None;
    }
    let domain_start = local + 1;
    let domain_end = run(chars, domain_start, |value| {
        value.is_alphanumeric() || "_.-".contains(value)
    });
    (domain_start + 1..domain_end).rev().find_map(|dot| {
        if chars[dot] != '.' {
            return None;
        }
        let letters_end = run(chars, dot + 1, |value| value.is_ascii_alphabetic());
        let closed = !word(chars.get(letters_end));
        (letters_end - (dot + 1) >= MIN_TLD_LETTERS && closed).then_some(letters_end)
    })
}

fn number_end(chars: &[char], start: usize) -> Option<usize> {
    if !boundary(chars, start) || !chars[start].is_ascii_digit() {
        return None;
    }
    let longest = run(chars, start + 1, |value| {
        value.is_ascii_digit() || ".,:/%-".contains(value)
    });
    (start + 1..=longest)
        .rev()
        .find(|end| boundary(chars, *end))
}

/// Every anchor in `text`, as the set of matched spans.
pub(crate) fn protected_anchors(text: &str) -> HashSet<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut anchors = HashSet::new();
    let mut index = 0;
    while index < chars.len() {
        let end = url_end(&chars, index)
            .or_else(|| email_end(&chars, index))
            .or_else(|| number_end(&chars, index));
        match end {
            Some(end) => {
                anchors.insert(chars[index..end].iter().collect());
                index = end;
            }
            None => index += 1,
        }
    }
    anchors
}

/// Whether a teacher's generic rewrite may stand as the source of `target`:
/// it differs from the target and repeats every anchor the target holds.
/// How long it may be is not judged by a ratio nobody stated: the
/// independent review decides whether the pair is faithful.
pub(crate) fn valid_source(target: &str, source: &str) -> bool {
    let target = target.trim();
    let source = source.trim();
    if source.is_empty() || source == target {
        return false;
    }
    protected_anchors(target).is_subset(&protected_anchors(source))
}
