use super::*;

pub(crate) fn write_json(out: &mut String, value: &Value, depth: usize) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(flag) => out.push_str(if *flag { "true" } else { "false" }),
        Value::Number(number) => match number.as_f64() {
            Some(float) if number.is_f64() => out.push_str(&float_repr(float)),
            _ => out.push_str(&number.to_string()),
        },
        Value::String(string) => write_json_string(out, string),
        Value::Array(items) => {
            if items.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push_str("[\n");
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push_str(",\n");
                }
                indent(out, depth + 1);
                write_json(out, item, depth + 1);
            }
            out.push('\n');
            indent(out, depth);
            out.push(']');
        }
        Value::Object(map) => {
            if map.is_empty() {
                out.push_str("{}");
                return;
            }
            out.push_str("{\n");
            for (index, (key, item)) in map.iter().enumerate() {
                if index > 0 {
                    out.push_str(",\n");
                }
                indent(out, depth + 1);
                write_json_string(out, key);
                out.push_str(": ");
                write_json(out, item, depth + 1);
            }
            out.push('\n');
            indent(out, depth);
            out.push('}');
        }
    }
}

pub(crate) fn indent(out: &mut String, depth: usize) {
    for _ in 0..depth * 2 {
        out.push(' ');
    }
}

pub(crate) fn write_json_string(out: &mut String, value: &str) {
    out.push('"');
    for character in value.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            character if (character as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", character as u32);
            }
            character if (character as u32) < 0x7f => out.push(character),
            character => {
                let code = character as u32;
                if code > 0xFFFF {
                    let value = code - 0x10000;
                    let high = 0xD800 + (value >> 10);
                    let low = 0xDC00 + (value & 0x3FF);
                    let _ = write!(out, "\\u{high:04x}\\u{low:04x}");
                } else {
                    let _ = write!(out, "\\u{code:04x}");
                }
            }
        }
    }
    out.push('"');
}

/// Python's `repr()` of a value. `info` interpolates the class list, and
/// `['agent', 'data']` is what the README shows.
pub(crate) fn repr(value: &Value) -> String {
    match value {
        Value::Null => "None".to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::Number(number) => match number.as_f64() {
            Some(float) if number.is_f64() => float_repr(float),
            _ => number.to_string(),
        },
        Value::String(string) => repr_str(string),
        Value::Array(items) => format!(
            "[{}]",
            items.iter().map(repr).collect::<Vec<_>>().join(", ")
        ),
        Value::Object(map) => format!(
            "{{{}}}",
            map.iter()
                .map(|(key, item)| format!("{}: {}", repr_str(key), repr(item)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

pub(crate) fn repr_str(value: &str) -> String {
    let quote = if value.contains('\'') && !value.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(value.len() + 2);
    out.push(quote);
    for character in value.chars() {
        match character {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            character if character == quote => {
                out.push('\\');
                out.push(character);
            }
            character => out.push(character),
        }
    }
    out.push(quote);
    out
}

// ------------------------------------------------------------ the arguments

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    Flag,
    Text,
    Int,
    Float,
}

pub(crate) struct Opt {
    pub(crate) flag: &'static str,
    pub(crate) metavar: &'static str,
    pub(crate) kind: Kind,
    pub(crate) required: bool,
    /// Non-zero puts this option in a mutually exclusive group with every
    /// other option carrying the same number.
    pub(crate) group: u8,
    pub(crate) help: String,
}

pub(crate) struct Positional {
    pub(crate) name: &'static str,
    pub(crate) help: String,
}

pub(crate) struct Spec {
    pub(crate) name: &'static str,
    pub(crate) help: String,
    pub(crate) description: Option<String>,
    pub(crate) positionals: Vec<Positional>,
    pub(crate) opts: Vec<Opt>,
}

pub(crate) fn option(flag: &'static str, metavar: &'static str, kind: Kind, help: String) -> Opt {
    Opt {
        flag,
        metavar,
        kind,
        required: false,
        group: 0,
        help,
    }
}

pub(crate) fn required(flag: &'static str, metavar: &'static str, kind: Kind, help: String) -> Opt {
    Opt {
        flag,
        metavar,
        kind,
        required: true,
        group: 0,
        help,
    }
}
