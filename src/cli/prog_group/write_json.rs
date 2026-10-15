use super::*;

/// One level of Python's `json.dumps(value, indent=2)`.
const INDENT: &str = "  ";

/// Write `value` the way Python's `json.dumps(value, indent=2)` does, with
/// `outer` the indentation of the line it starts on.
pub(crate) fn write_json(out: &mut String, value: &Value, outer: &str) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(flag) => out.push_str(&flag.to_string()),
        Value::Number(number) => match number.as_f64() {
            Some(float) if number.is_f64() => out.push_str(&float_repr(float)),
            _ => out.push_str(&number.to_string()),
        },
        Value::String(string) => write_json_string(out, string),
        Value::Array(items) if items.is_empty() => out.push_str("[]"),
        Value::Array(items) => {
            let inner = format!("{outer}{INDENT}");
            out.push_str("[\n");
            let mut separator = "";
            for item in items {
                out.push_str(separator);
                out.push_str(&inner);
                write_json(out, item, &inner);
                separator = ",\n";
            }
            out.push('\n');
            out.push_str(outer);
            out.push(']');
        }
        Value::Object(map) if map.is_empty() => out.push_str("{}"),
        Value::Object(map) => {
            let inner = format!("{outer}{INDENT}");
            out.push_str("{\n");
            let mut separator = "";
            for (key, item) in map {
                out.push_str(separator);
                out.push_str(&inner);
                write_json_string(out, key);
                out.push_str(": ");
                write_json(out, item, &inner);
                separator = ",\n";
            }
            out.push('\n');
            out.push_str(outer);
            out.push('}');
        }
    }
}

/// A JSON string as Python's `ensure_ascii` writes it: printable ASCII as
/// itself, the short escapes JSON names, and every other character as the
/// `\uXXXX` escape of each of its UTF-16 code units.
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
            character if character == ' ' || character.is_ascii_graphic() => out.push(character),
            character => {
                for unit in character.to_string().encode_utf16() {
                    out.push_str(&format!("\\u{unit:04x}"));
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

/// Python's `repr()` of a string: single quotes unless the text holds a
/// single quote and no double quote.
pub(crate) fn repr_str(value: &str) -> String {
    let quote = match value.contains('\'') && !value.contains('"') {
        true => '"',
        false => '\'',
    };
    let mut out = String::new();
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
    /// The mutually exclusive group this option belongs to, by name: at most
    /// one option of a group may be given.
    pub(crate) group: Option<&'static str>,
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
        group: None,
        help,
    }
}

pub(crate) fn required(flag: &'static str, metavar: &'static str, kind: Kind, help: String) -> Opt {
    Opt {
        flag,
        metavar,
        kind,
        required: true,
        group: None,
        help,
    }
}
