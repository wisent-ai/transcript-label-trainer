//! Transcript Label Trainer: trains small local classifiers over the aspect
//! labels the Transcript Lake owns, and never writes to the lake itself.
mod autolabel;
mod brama;
mod cli;
mod evaluate;
mod goal;
#[cfg(feature = "hf")]
mod hf;
mod humanizer;
mod jobs;
mod lake;
mod lifecycle;
mod model;
mod placement;
mod stado;
mod util;

use std::process::ExitCode;

fn main() -> ExitCode {
    match cli::run(std::env::args().skip(1).collect()) {
        Ok(code) => ExitCode::from(u8::try_from(code).unwrap_or(1)),
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}
