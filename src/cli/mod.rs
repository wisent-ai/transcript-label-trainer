//! Command-line interface: train, run, infer, evaluate, info, autolabel.
//!
//! The parser is hand-rolled rather than pulled from a crate, because the
//! surface it has to reproduce is not "some CLI" but exactly the one argparse
//! produced: the same flags, the same defaults, the same `--help` bodies, the
//! same `error:` sentences and the same exit statuses. Those strings are
//! quoted by the README and parsed by callers, so they are the contract.

use serde_json::Value;

use crate::util::{float_repr, json_text, json_truthy, Error, Result, TrainFailure};
use crate::{
    autolabel, brama, corpus, discover, evaluate, goal, gui, jobs, model, onboarding, placement,
    stado,
};

/// The status of a command that did what it was asked.
pub(crate) const EXIT_SUCCESS: i32 = 0; // https://pubs.opengroup.org/onlinepubs/9799919799/functions/exit.html EXIT_SUCCESS
/// The status of a command that failed.
pub(crate) const EXIT_FAILURE: i32 = 1; // https://pubs.opengroup.org/onlinepubs/9799919799/functions/exit.html EXIT_FAILURE
/// The status of a command whose prerequisite is missing, such as too little
/// labeled data to train on.
pub(crate) const EXIT_PREREQUISITE_MISSING: i32 = 2; // https://github.com/wisent-ai/transcript-label-trainer/blob/main/docs/guide/models/humanizer.md "exit 2 means a prerequisite is missing"

/// `println!` that does not panic when the reader has closed the pipe.
///
/// `transcript-label-trainer info | head` is a normal thing to type, and a
/// Rust panic is not an answer to it; a CLI that has nowhere left to write
/// simply stops writing. Any other failure to write is a failure, named.
macro_rules! outln {
    ($($arg:tt)*) => {{
        use std::io::Write as _;
        match writeln!(std::io::stdout(), $($arg)*) {
            Err(error) if error.kind() != std::io::ErrorKind::BrokenPipe => {
                panic!("could not write to standard output: {error}")
            }
            _ => {}
        }
    }};
}

mod build_specs_group;
mod prog_group;
mod release_group;

pub use build_specs_group::*;
pub use prog_group::*;
pub use release_group::*;

mod specs_corpus;
mod specs_goal;
mod specs_humanizer;
mod specs_humanizer_served;
mod specs_models;

pub use specs_corpus::*;
pub use specs_goal::*;
pub use specs_humanizer::*;
pub use specs_humanizer_served::*;
pub use specs_models::*;
