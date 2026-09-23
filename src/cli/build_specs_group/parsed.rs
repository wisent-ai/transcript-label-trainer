use super::*;

#[derive(Default)]
pub(crate) struct Parsed {
    pub(crate) flags: Vec<&'static str>,
    pub(crate) texts: Vec<(&'static str, String)>,
    pub(crate) ints: Vec<(&'static str, i64)>,
    pub(crate) floats: Vec<(&'static str, f64)>,
    pub(crate) positionals: Vec<String>,
}

impl Parsed {
    pub(crate) fn flag(&self, name: &str) -> bool {
        self.flags.iter().any(|seen| *seen == name)
    }

    pub(crate) fn text(&self, name: &str) -> Option<&str> {
        self.texts
            .iter()
            .find(|(key, _)| *key == name)
            .map(|(_, value)| value.as_str())
    }

    pub(crate) fn int(&self, name: &str) -> Option<i64> {
        self.ints
            .iter()
            .find(|(key, _)| *key == name)
            .map(|(_, value)| *value)
    }

    pub(crate) fn float(&self, name: &str) -> Option<f64> {
        self.floats
            .iter()
            .find(|(key, _)| *key == name)
            .map(|(_, value)| *value)
    }

    pub(crate) fn positional(&self, index: usize) -> &str {
        self.positionals
            .get(index)
            .map(String::as_str)
            .unwrap_or("")
    }

    pub(crate) fn has(&self, opt: &Opt) -> bool {
        match opt.kind {
            Kind::Flag => self.flag(opt.flag),
            Kind::Text => self.text(opt.flag).is_some(),
            Kind::Int => self.int(opt.flag).is_some(),
            Kind::Float => self.float(opt.flag).is_some(),
        }
    }
}

pub(crate) enum Outcome {
    Parsed(Parsed),
    Help,
    Error(String),
}

pub(crate) fn parse_sub(spec: &Spec, argv: &[String]) -> Outcome {
    let mut parsed = Parsed::default();
    let mut extras: Vec<String> = Vec::new();
    let mut in_group: Vec<(u8, &'static str)> = Vec::new();
    let mut index = 0;

    while index < argv.len() {
        let arg = argv[index].as_str();
        if arg == "-h" || arg == "--help" {
            return Outcome::Help;
        }
        let (name, inline) = match arg.split_once('=') {
            Some((name, value)) if name.starts_with("--") => (name, Some(value.to_string())),
            _ => (arg, None),
        };
        let Some(opt) = spec.opts.iter().find(|opt| opt.flag == name) else {
            if looks_like_option(arg) {
                extras.push(arg.to_string());
            } else {
                parsed.positionals.push(arg.to_string());
            }
            index += 1;
            continue;
        };

        if opt.kind == Kind::Flag {
            if let Some(value) = inline {
                return Outcome::Error(format!(
                    "argument {}: ignored explicit argument '{value}'",
                    opt.flag
                ));
            }
            parsed.flags.push(opt.flag);
            index += 1;
        } else {
            let value = match inline {
                Some(value) => {
                    index += 1;
                    value
                }
                None => match argv.get(index + 1) {
                    Some(next) if !looks_like_option(next) => {
                        index += 2;
                        next.clone()
                    }
                    _ => {
                        return Outcome::Error(format!(
                            "argument {}: expected one argument",
                            opt.flag
                        ));
                    }
                },
            };
            match opt.kind {
                Kind::Text => parsed.texts.push((opt.flag, value)),
                Kind::Int => match value.trim().parse::<i64>() {
                    Ok(number) => parsed.ints.push((opt.flag, number)),
                    Err(_) => {
                        return Outcome::Error(format!(
                            "argument {}: invalid int value: '{value}'",
                            opt.flag
                        ));
                    }
                },
                Kind::Float => match value.trim().parse::<f64>() {
                    Ok(number) => parsed.floats.push((opt.flag, number)),
                    Err(_) => {
                        return Outcome::Error(format!(
                            "argument {}: invalid float value: '{value}'",
                            opt.flag
                        ));
                    }
                },
                Kind::Flag => {}
            }
        }

        if opt.group != 0 {
            if let Some((_, other)) = in_group
                .iter()
                .find(|(group, other)| *group == opt.group && *other != opt.flag)
            {
                return Outcome::Error(format!(
                    "argument {}: not allowed with argument {other}",
                    opt.flag
                ));
            }
            in_group.push((opt.group, opt.flag));
        }
    }

    if parsed.positionals.len() > spec.positionals.len() {
        extras.extend(parsed.positionals.split_off(spec.positionals.len()));
    }
    if !extras.is_empty() {
        return Outcome::Error(format!("unrecognized arguments: {}", extras.join(" ")));
    }

    let mut missing: Vec<&str> = Vec::new();
    for (index, positional) in spec.positionals.iter().enumerate() {
        if parsed.positionals.len() <= index {
            missing.push(positional.name);
        }
    }
    for opt in &spec.opts {
        if opt.required && !parsed.has(opt) {
            missing.push(opt.flag);
        }
    }
    if !missing.is_empty() {
        return Outcome::Error(format!(
            "the following arguments are required: {}",
            missing.join(", ")
        ));
    }
    Outcome::Parsed(parsed)
}

/// A token argparse would read as an option rather than as a value: it starts
/// with a dash, is longer than a bare `-`, and is not a negative number.
pub(crate) fn looks_like_option(arg: &str) -> bool {
    if !arg.starts_with('-') || arg.len() < 2 {
        return false;
    }
    arg[1..].parse::<f64>().is_err()
}

// ----------------------------------------------------------------- the help

pub(crate) struct Action {
    pub(crate) invocation: String,
    pub(crate) help: Option<String>,
    pub(crate) indent: usize,
}

pub(crate) fn top_usage_parts(specs: &[Spec]) -> (Vec<String>, Vec<String>) {
    let optionals = vec![
        "[-h]".to_string(),
        "[-V]".to_string(),
        "[--training-root PATH]".to_string(),
        "[--storage-root PATH]".to_string(),
    ];
    (optionals, vec![format!("{} ...", choices_metavar(specs))])
}

pub(crate) fn choices_metavar(specs: &[Spec]) -> String {
    format!(
        "{{{}}}",
        specs
            .iter()
            .map(|spec| spec.name)
            .collect::<Vec<_>>()
            .join(",")
    )
}

pub(crate) fn top_help(specs: &[Spec]) -> String {
    let (optionals, positionals) = top_usage_parts(specs);
    let mut choices = vec![Action {
        invocation: choices_metavar(specs),
        help: None,
        indent: 2,
    }];
    for spec in specs {
        choices.push(Action {
            invocation: spec.name.to_string(),
            help: Some(spec.help.clone()),
            indent: 4,
        });
    }
    let options = vec![
        Action {
            invocation: "-h, --help".to_string(),
            help: Some(HELP_HELP.to_string()),
            indent: 2,
        },
        Action {
            invocation: "-V, --version".to_string(),
            help: Some(VERSION_HELP.to_string()),
            indent: 2,
        },
        Action {
            invocation: "--training-root PATH".to_string(),
            help: Some(TRAINING_ROOT_HELP.to_string()),
            indent: 2,
        },
        Action {
            invocation: "--storage-root PATH".to_string(),
            help: Some(STORAGE_ROOT_HELP.to_string()),
            indent: 2,
        },
    ];
    assemble(
        PROG,
        &optionals,
        &positionals,
        Some(TOP_DESCRIPTION),
        &choices,
        &options,
    )
}
