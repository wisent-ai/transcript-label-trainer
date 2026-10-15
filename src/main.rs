//! Transcript Label Trainer: trains small local classifiers over the aspect
//! labels the Transcript Lake owns, and never writes to the lake itself.
mod autolabel;
mod brama;
mod cli;
mod corpus;
mod discover;
mod evaluate;
mod goal;
mod gui;
#[cfg(feature = "hf")]
mod hf;
mod hub;
mod humanizer;
mod jobs;
mod lake;
mod lifecycle;
mod model;
mod onboarding;
mod placement;
mod serving;
mod stado;
mod util;

use std::process::ExitCode;

fn main() -> ExitCode {
    // The trainer's local roots and its onboarding state live under the
    // account's home directory; a process without one is refused here, before
    // either is placed somewhere else.
    if !util::home_is_named() {
        eprintln!("error: HOME is unset or empty; the trainer needs the account's home directory");
        return ExitCode::FAILURE;
    }
    let mut arguments = std::env::args();
    // The first argument is the program's own name, not an argument to it.
    arguments.next();
    match cli::run(arguments.collect()) {
        Ok(code) => match u8::try_from(code) {
            Ok(status) => ExitCode::from(status),
            Err(error) => {
                eprintln!(
                    "error: the command answered exit status {code}, which a process cannot exit with: {error}"
                );
                ExitCode::FAILURE
            }
        },
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}
