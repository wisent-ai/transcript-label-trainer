//! Error type and the small helpers every module shares. One operator-facing
//! sentence per failure; `main` prints it and exits non-zero.
use std::fmt;
use std::path::PathBuf;

#[derive(Debug)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

impl From<String> for Error {
    fn from(value: String) -> Self {
        Error(value)
    }
}

impl From<&str> for Error {
    fn from(value: &str) -> Self {
        Error(value.to_string())
    }
}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Error(value.to_string())
    }
}

impl From<serde_json::Error> for Error {
    fn from(value: serde_json::Error) -> Self {
        Error(value.to_string())
    }
}

impl From<serde_yaml::Error> for Error {
    fn from(value: serde_yaml::Error) -> Self {
        Error(value.to_string())
    }
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Fail with a formatted message.
#[macro_export]
macro_rules! bail {
    ($($arg:tt)*) => {
        return Err($crate::util::Error(format!($($arg)*)))
    };
}

/// `$HOME`, or `.` when the environment has none.
pub fn home_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

/// Current instant as an ISO-8601 UTC string, second precision, matching the
/// label store's own timestamp shape.
pub fn now_iso() -> String {
    chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string()
}

/// How a training run failed.
///
/// The Python CLI exited 2 when the lake simply did not hold enough labeled
/// sessions yet, and 1 for every other failure — a documented difference
/// between "nothing is wrong, go label more" and "this run is broken", which
/// scripts around `train` and `run` read. The split survives the port as a
/// type instead of an exception class.
#[derive(Debug)]
pub enum TrainFailure {
    /// Too few labeled sessions, or too few distinct values, to fit anything.
    NotEnoughData(String),
    /// Anything else: a bad spec, a missing artifact, an unreachable lake.
    Failed(Error),
}

impl fmt::Display for TrainFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TrainFailure::NotEnoughData(message) => f.write_str(message),
            TrainFailure::Failed(error) => fmt::Display::fmt(error, f),
        }
    }
}

impl std::error::Error for TrainFailure {}

impl From<Error> for TrainFailure {
    fn from(value: Error) -> Self {
        TrainFailure::Failed(value)
    }
}

impl From<String> for TrainFailure {
    fn from(value: String) -> Self {
        TrainFailure::Failed(Error(value))
    }
}

impl From<&str> for TrainFailure {
    fn from(value: &str) -> Self {
        TrainFailure::Failed(Error(value.to_string()))
    }
}

impl From<std::io::Error> for TrainFailure {
    fn from(value: std::io::Error) -> Self {
        TrainFailure::Failed(Error(value.to_string()))
    }
}

impl From<serde_json::Error> for TrainFailure {
    fn from(value: serde_json::Error) -> Self {
        TrainFailure::Failed(Error(value.to_string()))
    }
}

impl From<serde_yaml::Error> for TrainFailure {
    fn from(value: serde_yaml::Error) -> Self {
        TrainFailure::Failed(Error(value.to_string()))
    }
}

/// A JSON scalar rendered the way the Python original's `str()` rendered it.
///
/// Registry values, label fields and metric fields all reach the operator
/// through interpolation, and the printed shapes are what the README and the
/// examples quote — including `None` for a missing value.
pub fn json_text(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => "None".to_string(),
        serde_json::Value::Bool(true) => "True".to_string(),
        serde_json::Value::Bool(false) => "False".to_string(),
        serde_json::Value::String(text) => text.clone(),
        serde_json::Value::Number(number) => match number.as_f64() {
            Some(float) if number.is_f64() => float_repr(float),
            _ => number.to_string(),
        },
        other => other.to_string(),
    }
}

/// Python truthiness of a JSON value: null, false, zero, and every empty
/// container are false. The porting targets test values with `if value:`.
pub fn json_truthy(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Null => false,
        serde_json::Value::Bool(flag) => *flag,
        serde_json::Value::Number(number) => number.as_f64().is_some_and(|n| n != 0.0),
        serde_json::Value::String(text) => !text.is_empty(),
        serde_json::Value::Array(items) => !items.is_empty(),
        serde_json::Value::Object(map) => !map.is_empty(),
    }
}

/// `repr()` of a Python float: shortest round-trip digits, fixed notation
/// while the decimal point sits in `(-4, 16]`, and a two-digit signed
/// exponent otherwise. `2e-05`, not Rust's `2e-5`; `3.0`, not `3`.
///
/// Metrics files and the JSON the CLI prints carry floats, and the byte shape
/// of those files is the compatibility contract with everything already on
/// disk, so the formatting cannot come from Rust's default.
pub fn float_repr(value: f64) -> String {
    if value.is_nan() {
        return "nan".to_string();
    }
    if value.is_infinite() {
        return if value < 0.0 {
            "-inf".to_string()
        } else {
            "inf".to_string()
        };
    }
    // `{:e}` gives the shortest round-tripping digits as `<mantissa>e<exp>`.
    let scientific = format!("{value:e}");
    let (mantissa, exponent) = match scientific.split_once('e') {
        Some(parts) => parts,
        None => return scientific,
    };
    let exponent: i32 = exponent.parse().unwrap_or(0);
    let (sign, mantissa) = match mantissa.strip_prefix('-') {
        Some(rest) => ("-", rest),
        None => ("", mantissa),
    };
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let digits = digits.trim_end_matches('0');
    let digits = if digits.is_empty() { "0" } else { digits };
    // Position of the decimal point: value == 0.<digits> * 10^point.
    let point = exponent + 1;
    if point > -4 && point <= 16 {
        let length = digits.len() as i32;
        let body = if point <= 0 {
            format!("0.{}{}", "0".repeat((-point) as usize), digits)
        } else if point >= length {
            format!("{}{}.0", digits, "0".repeat((point - length) as usize))
        } else {
            format!(
                "{}.{}",
                &digits[..point as usize],
                &digits[point as usize..]
            )
        };
        return format!("{sign}{body}");
    }
    let body = if digits.len() > 1 {
        format!("{}.{}", &digits[..1], &digits[1..])
    } else {
        digits.to_string()
    };
    let exponent = point - 1;
    let exponent_sign = if exponent < 0 { '-' } else { '+' };
    format!("{sign}{body}e{exponent_sign}{:02}", exponent.abs())
}
