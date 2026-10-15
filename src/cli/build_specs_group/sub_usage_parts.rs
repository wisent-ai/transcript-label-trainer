use super::*;

// The help layout reproduces Python's argparse HelpFormatter, whose output is
// the contract; each measure below is the one argparse uses.

/// Columns each nesting level of the help is indented by.
pub(crate) const INDENT_INCREMENT: usize = 2; // https://github.com/python/cpython/blob/3.12/Lib/argparse.py HelpFormatter.__init__(indent_increment=2)
/// Columns between the widest invocation and its help, and the gap argparse
/// pads an invocation with before its help.
const HELP_GAP: usize = 2; // https://github.com/python/cpython/blob/3.12/Lib/argparse.py add_argument: self._action_max_length + 2; _format_action: '%*s%-*s  '
/// The narrowest column argparse wraps help text in.
const MIN_HELP_WIDTH: usize = 11; // https://github.com/python/cpython/blob/3.12/Lib/argparse.py _format_action: max(self._width - help_position, 11)
/// The share of the width a program name may take before the usage folds
/// its parts under the name instead of beside it.
const USAGE_FOLD_SHARE: f64 = 0.75; // https://github.com/python/cpython/blob/3.12/Lib/argparse.py _format_usage: len(prefix) + len(prog) <= 0.75 * text_width
/// What usage parts are joined with.
const PART_SEPARATOR: &str = " ";

pub(crate) fn sub_usage_parts(spec: &Spec) -> (Vec<String>, Vec<String>) {
    let mut optionals = vec!["[-h]".to_string()];
    let mut grouped: Vec<&str> = Vec::new();
    for opt in &spec.opts {
        match opt.group {
            Some(group) if grouped.contains(&group) => {}
            Some(group) => {
                grouped.push(group);
                let members: Vec<String> = spec
                    .opts
                    .iter()
                    .filter(|other| other.group == Some(group))
                    .map(bare_usage)
                    .collect();
                optionals.push(format!("[{}]", members.join(" | ")));
            }
            None if opt.required => optionals.push(bare_usage(opt)),
            None => optionals.push(format!("[{}]", bare_usage(opt))),
        }
    }
    let positionals = spec
        .positionals
        .iter()
        .map(|positional| positional.name.to_string())
        .collect();
    (optionals, positionals)
}

pub(crate) fn bare_usage(opt: &Opt) -> String {
    match opt.kind {
        Kind::Flag => opt.flag.to_string(),
        _ => format!("{} {}", opt.flag, opt.metavar),
    }
}

pub(crate) fn sub_help(spec: &Spec) -> String {
    let prog = format!("{PROG} {}", spec.name);
    let (optionals, positional_parts) = sub_usage_parts(spec);
    let positionals: Vec<Action> = spec
        .positionals
        .iter()
        .map(|positional| Action {
            invocation: positional.name.to_string(),
            help: Some(positional.help.clone()),
            indent: INDENT_INCREMENT,
        })
        .collect();
    let mut options = vec![Action {
        invocation: "-h, --help".to_string(),
        help: Some(HELP_HELP.to_string()),
        indent: INDENT_INCREMENT,
    }];
    for opt in &spec.opts {
        options.push(Action {
            invocation: bare_usage(opt),
            help: Some(opt.help.clone()),
            indent: INDENT_INCREMENT,
        });
    }
    assemble(
        &prog,
        &optionals,
        &positional_parts,
        spec.description.as_deref(),
        &positionals,
        &options,
    )
}

/// One titled section of actions, aligned on `help_position`.
fn push_section(out: &mut String, title: &str, actions: &[Action], help_position: usize) {
    out.push('\n');
    out.push_str(title);
    out.push('\n');
    for action in actions {
        format_action(action, help_position, out);
    }
}

pub(crate) fn assemble(
    prog: &str,
    optionals: &[String],
    positional_parts: &[String],
    description: Option<&str>,
    positionals: &[Action],
    options: &[Action],
) -> String {
    let mut out = format_usage(prog, optionals, positional_parts);
    // Help starts past the widest invocation, never past argparse's limit;
    // with no actions there is no column to align.
    let help_position = positionals
        .iter()
        .chain(options.iter())
        .map(|action| action.invocation.chars().count() + action.indent)
        .max()
        .map(|widest| (widest + HELP_GAP).min(MAX_HELP_POSITION));

    if let Some(description) = description {
        out.push('\n');
        for line in wrap(description, WIDTH) {
            out.push_str(&line);
            out.push('\n');
        }
    }
    if let Some(help_position) = help_position {
        if !positionals.is_empty() {
            push_section(
                &mut out,
                "positional arguments:",
                positionals,
                help_position,
            );
        }
        if !options.is_empty() {
            push_section(&mut out, "options:", options, help_position);
        }
    }
    out
}

pub(crate) fn format_action(action: &Action, help_position: usize, out: &mut String) {
    let header_length = action.invocation.chars().count();
    let indent = " ".repeat(action.indent);
    let Some(help) = action
        .help
        .as_deref()
        .filter(|help| !help.trim().is_empty())
    else {
        out.push_str(&format!("{indent}{}\n", action.invocation));
        return;
    };
    let action_width = help_position.saturating_sub(action.indent + HELP_GAP);
    let help_width = WIDTH.saturating_sub(help_position).max(MIN_HELP_WIDTH);
    let help_indent = " ".repeat(help_position);

    out.push_str(&indent);
    out.push_str(&action.invocation);
    match action_width.checked_sub(header_length) {
        Some(padding) => out.push_str(&" ".repeat(padding + HELP_GAP)),
        None => {
            out.push('\n');
            out.push_str(&help_indent);
        }
    }
    let mut lines = wrap(help, help_width).into_iter();
    if let Some(first) = lines.next() {
        out.push_str(&first);
    }
    out.push('\n');
    for line in lines {
        out.push_str(&format!("{help_indent}{line}\n"));
    }
}

/// argparse's usage line, wrapping included: one line while it fits, then the
/// option parts folded under the program name.
pub(crate) fn format_usage(prog: &str, optionals: &[String], positionals: &[String]) -> String {
    let prefix = "usage: ";
    let single = std::iter::once(prog.to_string())
        .chain(optionals.iter().cloned())
        .chain(positionals.iter().cloned())
        .collect::<Vec<_>>()
        .join(PART_SEPARATOR);
    if prefix.len() + single.chars().count() <= WIDTH {
        return format!("{prefix}{single}\n");
    }

    let prog_length = prog.chars().count();
    let lines: Vec<String> =
        if (prefix.len() + prog_length) as f64 <= USAGE_FOLD_SHARE * WIDTH as f64 {
            let indent = " ".repeat(prefix.len() + prog_length + PART_SEPARATOR.len());
            let head = |parts: &[String]| {
                std::iter::once(prog.to_string())
                    .chain(parts.iter().cloned())
                    .collect::<Vec<String>>()
            };
            match (optionals.is_empty(), positionals.is_empty()) {
                (false, _) => {
                    let mut lines = usage_lines(&head(optionals), &indent, Some(prefix));
                    lines.extend(usage_lines(positionals, &indent, None));
                    lines
                }
                (true, false) => usage_lines(&head(positionals), &indent, Some(prefix)),
                (true, true) => vec![prog.to_string()],
            }
        } else {
            let indent = " ".repeat(prefix.len());
            let parts: Vec<String> = optionals.iter().chain(positionals).cloned().collect();
            let mut lines = usage_lines(&parts, &indent, None);
            // argparse keeps optionals and positionals on separate line groups
            // once their joint wrap takes more than one line.
            if matches!(lines.as_slice(), [_, _, ..]) {
                lines = usage_lines(optionals, &indent, None);
                lines.extend(usage_lines(positionals, &indent, None));
            }
            std::iter::once(prog.to_string()).chain(lines).collect()
        };
    format!("{prefix}{}\n", lines.join("\n"))
}

pub(crate) fn usage_lines(parts: &[String], indent: &str, prefix: Option<&str>) -> Vec<String> {
    let indent_length = indent.chars().count();
    let separator = PART_SEPARATOR.len();
    let mut lines: Vec<String> = Vec::new();
    let mut line: Vec<&str> = Vec::new();
    let mut line_length = match prefix {
        Some(prefix) => prefix.chars().count().saturating_sub(separator),
        None => indent_length.saturating_sub(separator),
    };
    for part in parts {
        let part_length = part.chars().count();
        if line_length + separator + part_length > WIDTH && !line.is_empty() {
            lines.push(format!("{indent}{}", line.join(PART_SEPARATOR)));
            line.clear();
            line_length = indent_length.saturating_sub(separator);
        }
        line.push(part);
        line_length += separator + part_length;
    }
    if !line.is_empty() {
        lines.push(format!("{indent}{}", line.join(PART_SEPARATOR)));
    }
    if prefix.is_some() {
        if let Some(first) = lines.first_mut() {
            *first = first.chars().skip(indent_length).collect();
        }
    }
    lines
}
