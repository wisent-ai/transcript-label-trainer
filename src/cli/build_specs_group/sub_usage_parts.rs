use super::*;

pub(crate) fn sub_usage_parts(spec: &Spec) -> (Vec<String>, Vec<String>) {
    let mut optionals = vec!["[-h]".to_string()];
    let mut grouped: Vec<u8> = Vec::new();
    for opt in &spec.opts {
        if opt.group != 0 {
            if grouped.contains(&opt.group) {
                continue;
            }
            grouped.push(opt.group);
            let members: Vec<String> = spec
                .opts
                .iter()
                .filter(|other| other.group == opt.group)
                .map(bare_usage)
                .collect();
            optionals.push(format!("[{}]", members.join(" | ")));
            continue;
        }
        let bare = bare_usage(opt);
        optionals.push(if opt.required {
            bare
        } else {
            format!("[{bare}]")
        });
    }
    let positionals = spec
        .positionals
        .iter()
        .map(|positional| positional.name.to_string())
        .collect();
    (optionals, positionals)
}

pub(crate) fn bare_usage(opt: &Opt) -> String {
    if opt.kind == Kind::Flag {
        opt.flag.to_string()
    } else {
        format!("{} {}", opt.flag, opt.metavar)
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
            indent: 2,
        })
        .collect();
    let mut options = vec![Action {
        invocation: "-h, --help".to_string(),
        help: Some(HELP_HELP.to_string()),
        indent: 2,
    }];
    for opt in &spec.opts {
        options.push(Action {
            invocation: bare_usage(opt),
            help: Some(opt.help.clone()),
            indent: 2,
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

pub(crate) fn assemble(
    prog: &str,
    optionals: &[String],
    positional_parts: &[String],
    description: Option<&str>,
    positionals: &[Action],
    options: &[Action],
) -> String {
    let mut out = format_usage(prog, optionals, positional_parts);
    let action_max_length = positionals
        .iter()
        .chain(options.iter())
        .map(|action| action.invocation.chars().count() + action.indent)
        .max()
        .unwrap_or(0);
    let help_position = (action_max_length + 2).min(MAX_HELP_POSITION);

    if let Some(description) = description {
        out.push('\n');
        for line in wrap(description, WIDTH) {
            out.push_str(&line);
            out.push('\n');
        }
    }
    if !positionals.is_empty() {
        out.push('\n');
        out.push_str("positional arguments:\n");
        for action in positionals {
            format_action(action, help_position, &mut out);
        }
    }
    if !options.is_empty() {
        out.push('\n');
        out.push_str("options:\n");
        for action in options {
            format_action(action, help_position, &mut out);
        }
    }
    out
}

pub(crate) fn format_action(action: &Action, help_position: usize, out: &mut String) {
    let header_length = action.invocation.chars().count();
    let Some(help) = action
        .help
        .as_deref()
        .filter(|help| !help.trim().is_empty())
    else {
        push_spaces(out, action.indent);
        out.push_str(&action.invocation);
        out.push('\n');
        return;
    };
    let action_width = help_position.saturating_sub(action.indent + 2);
    let help_width = WIDTH.saturating_sub(help_position).max(11);
    let lines = wrap(help, help_width);

    push_spaces(out, action.indent);
    out.push_str(&action.invocation);
    if header_length <= action_width {
        push_spaces(out, action_width - header_length + 2);
    } else {
        out.push('\n');
        push_spaces(out, help_position);
    }
    let mut lines = lines.into_iter();
    if let Some(first) = lines.next() {
        out.push_str(&first);
    }
    out.push('\n');
    for line in lines {
        push_spaces(out, help_position);
        out.push_str(&line);
        out.push('\n');
    }
}

pub(crate) fn push_spaces(out: &mut String, count: usize) {
    for _ in 0..count {
        out.push(' ');
    }
}

/// argparse's usage line, wrapping included: one line while it fits, then the
/// option parts folded under the program name.
pub(crate) fn format_usage(prog: &str, optionals: &[String], positionals: &[String]) -> String {
    let prefix = "usage: ";
    let mut single = prog.to_string();
    for part in optionals.iter().chain(positionals.iter()) {
        single.push(' ');
        single.push_str(part);
    }
    if prefix.len() + single.chars().count() <= WIDTH {
        return format!("{prefix}{single}\n");
    }

    let prog_length = prog.chars().count();
    let lines: Vec<String> = if prefix.len() + prog_length <= WIDTH * 3 / 4 {
        let indent = " ".repeat(prefix.len() + prog_length + 1);
        if !optionals.is_empty() {
            let mut head = vec![prog.to_string()];
            head.extend(optionals.iter().cloned());
            let mut lines = usage_lines(&head, &indent, Some(prefix));
            lines.extend(usage_lines(positionals, &indent, None));
            lines
        } else if !positionals.is_empty() {
            let mut head = vec![prog.to_string()];
            head.extend(positionals.iter().cloned());
            usage_lines(&head, &indent, Some(prefix))
        } else {
            vec![prog.to_string()]
        }
    } else {
        let indent = " ".repeat(prefix.len());
        let mut parts = optionals.to_vec();
        parts.extend(positionals.iter().cloned());
        let mut lines = usage_lines(&parts, &indent, None);
        if lines.len() > 1 {
            lines = usage_lines(optionals, &indent, None);
            lines.extend(usage_lines(positionals, &indent, None));
        }
        let mut folded = vec![prog.to_string()];
        folded.extend(lines);
        folded
    };
    format!("{prefix}{}\n", lines.join("\n"))
}

pub(crate) fn usage_lines(parts: &[String], indent: &str, prefix: Option<&str>) -> Vec<String> {
    let indent_length = indent.chars().count();
    let mut lines: Vec<String> = Vec::new();
    let mut line: Vec<&str> = Vec::new();
    let mut line_length = match prefix {
        Some(prefix) => prefix.chars().count().saturating_sub(1),
        None => indent_length.saturating_sub(1),
    };
    for part in parts {
        let part_length = part.chars().count();
        if line_length + 1 + part_length > WIDTH && !line.is_empty() {
            lines.push(format!("{indent}{}", line.join(" ")));
            line.clear();
            line_length = indent_length.saturating_sub(1);
        }
        line.push(part);
        line_length += 1 + part_length;
    }
    if !line.is_empty() {
        lines.push(format!("{indent}{}", line.join(" ")));
    }
    if prefix.is_some() {
        if let Some(first) = lines.first_mut() {
            *first = first.chars().skip(indent_length).collect();
        }
    }
    lines
}
