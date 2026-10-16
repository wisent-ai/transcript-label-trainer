use super::*;

#[derive(Default)]
pub(crate) struct Parsed {
    pub(crate) flags: Vec<&'static str>,
    pub(crate) texts: Vec<(&'static str, String)>,
    pub(crate) ints: Vec<(&'static str, i64)>,
    pub(crate) floats: Vec<(&'static str, f64)>,
    /// Positional values in command-line order; `parse_sub` pairs each with
    /// its spec name once it knows every declared one is present.
    pub(crate) positionals: Vec<String>,
    pub(crate) named: Vec<(&'static str, String)>,
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

    /// The value of an option the command's spec marks `required`.
    /// `parse_sub` refuses a command line without it, so it is always
    /// present; asking for an option the spec does not require is a defect in
    /// the command that asks.
    pub(crate) fn required(&self, flag: &str) -> &str {
        match self.text(flag) {
            Some(value) => value,
            None => {
                panic!("the command reads `{flag}` as required, but its spec does not require it")
            }
        }
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

    /// The positional the command's spec declares as `name`. `parse_sub`
    /// refuses a command line that leaves a declared positional out, so a
    /// name the spec declares is always present; any other name is a defect
    /// in the command that asks for it.
    pub(crate) fn positional(&self, name: &str) -> &str {
        match self.named.iter().find(|(declared, _)| *declared == name) {
            Some((_, value)) => value,
            None => panic!("the command reads a positional `{name}` its spec does not declare"),
        }
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

/// Store one option's value as its kind reads it, or argparse's refusal.
fn store(parsed: &mut Parsed, opt: &Opt, value: String) -> std::result::Result<(), String> {
    match opt.kind {
        Kind::Text => parsed.texts.push((opt.flag, value)),
        Kind::Int => match value.trim().parse::<i64>() {
            Ok(number) => parsed.ints.push((opt.flag, number)),
            Err(error) => {
                return Err(format!(
                    "argument {}: invalid int value: '{value}' ({error})",
                    opt.flag
                ))
            }
        },
        Kind::Float => match value.trim().parse::<f64>() {
            Ok(number) => parsed.floats.push((opt.flag, number)),
            Err(error) => {
                return Err(format!(
                    "argument {}: invalid float value: '{value}' ({error})",
                    opt.flag
                ))
            }
        },
        Kind::Flag => parsed.flags.push(opt.flag),
    }
    Ok(())
}

pub(crate) fn parse_sub(spec: &Spec, argv: &[String]) -> Outcome {
    let mut parsed = Parsed::default();
    let mut extras: Vec<String> = Vec::new();
    let mut in_group: Vec<(&'static str, &'static str)> = Vec::new();
    let mut args = argv.iter().peekable();

    while let Some(arg) = args.next() {
        if arg == "-h" || arg == "--help" {
            return Outcome::Help;
        }
        let (name, inline) = match arg.split_once('=') {
            Some((name, value)) if name.starts_with("--") => (name, Some(value.to_string())),
            _ => (arg.as_str(), None),
        };
        let Some(opt) = spec.opts.iter().find(|opt| opt.flag == name) else {
            if looks_like_option(arg) {
                extras.push(arg.clone());
            } else {
                parsed.positionals.push(arg.clone());
            }
            continue;
        };

        let value = match (opt.kind, inline) {
            (Kind::Flag, Some(value)) => {
                return Outcome::Error(format!(
                    "argument {}: ignored explicit argument '{value}'",
                    opt.flag
                ));
            }
            (Kind::Flag, None) => String::new(),
            (_, Some(value)) => value,
            (_, None) => match args.next_if(|next| !looks_like_option(next)) {
                Some(next) => next.clone(),
                None => {
                    return Outcome::Error(format!("argument {}: expected one argument", opt.flag));
                }
            },
        };
        if let Err(refusal) = store(&mut parsed, opt, value) {
            return Outcome::Error(refusal);
        }

        if let Some(group) = opt.group {
            if let Some((_, other)) = in_group
                .iter()
                .find(|(seen, other)| *seen == group && *other != opt.flag)
            {
                return Outcome::Error(format!(
                    "argument {}: not allowed with argument {other}",
                    opt.flag
                ));
            }
            in_group.push((group, opt.flag));
        }
    }

    if parsed.positionals.len() > spec.positionals.len() {
        extras.extend(parsed.positionals.split_off(spec.positionals.len()));
    }
    if !extras.is_empty() {
        return Outcome::Error(format!("unrecognized arguments: {}", extras.join(" ")));
    }

    let missing: Vec<&str> = spec
        .positionals
        .iter()
        .skip(parsed.positionals.len())
        .map(|positional| positional.name)
        .chain(
            spec.opts
                .iter()
                .filter(|opt| opt.required && !parsed.has(opt))
                .map(|opt| opt.flag),
        )
        .collect();
    if !missing.is_empty() {
        return Outcome::Error(format!(
            "the following arguments are required: {}",
            missing.join(", ")
        ));
    }
    parsed.named = spec
        .positionals
        .iter()
        .map(|positional| positional.name)
        .zip(parsed.positionals.iter().cloned())
        .collect();
    Outcome::Parsed(parsed)
}

/// A token argparse would read as an option rather than as a value: it starts
/// with a dash, is longer than a bare `-`, and is not a negative number.
pub(crate) fn looks_like_option(arg: &str) -> bool {
    match arg.strip_prefix('-') {
        Some(rest) if !rest.is_empty() => rest.parse::<f64>().is_err(),
        _ => false,
    }
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
        indent: INDENT_INCREMENT,
    }];
    // argparse indents each subcommand one level under the choices line.
    for spec in specs {
        choices.push(Action {
            invocation: spec.name.to_string(),
            help: Some(spec.help.clone()),
            indent: INDENT_INCREMENT + INDENT_INCREMENT,
        });
    }
    let options = vec![
        Action {
            invocation: "-h, --help".to_string(),
            help: Some(HELP_HELP.to_string()),
            indent: INDENT_INCREMENT,
        },
        Action {
            invocation: "-V, --version".to_string(),
            help: Some(VERSION_HELP.to_string()),
            indent: INDENT_INCREMENT,
        },
        Action {
            invocation: "--training-root PATH".to_string(),
            help: Some(TRAINING_ROOT_HELP.to_string()),
            indent: INDENT_INCREMENT,
        },
        Action {
            invocation: "--storage-root PATH".to_string(),
            help: Some(STORAGE_ROOT_HELP.to_string()),
            indent: INDENT_INCREMENT,
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
