//! Command-line interface: train, run, infer, evaluate, info, autolabel.
//!
//! The parser is hand-rolled rather than pulled from a crate, because the
//! surface it has to reproduce is not "some CLI" but exactly the one argparse
//! produced: the same flags, the same defaults, the same `--help` bodies, the
//! same `error:` sentences and the same exit statuses. Those strings are
//! quoted by the README and parsed by callers, so they are the contract.

use std::fmt::Write as _;

use serde_json::Value;

use crate::util::{float_repr, json_text, json_truthy, Error, Result, TrainFailure};
use crate::{
    autolabel, brama, corpus, discover, evaluate, goal, gui, jobs, model, onboarding, placement,
    stado,
};

/// `println!` that does not panic when the reader has closed the pipe.
///
/// `transcript-label-trainer info | head` is a normal thing to type, and a
/// Rust panic is not an answer to it; a CLI that has nowhere left to write
/// simply stops writing.
macro_rules! outln {
    () => {{
        use std::io::Write as _;
        let _ = writeln!(std::io::stdout());
    }};
    ($($arg:tt)*) => {{
        use std::io::Write as _;
        let _ = writeln!(std::io::stdout(), $($arg)*);
    }};
}

mod prog_group;
mod build_specs_group;

pub use prog_group::*;
pub use build_specs_group::*;

mod specs_corpus;
mod specs_models;

pub use specs_corpus::*;
pub use specs_models::*;
