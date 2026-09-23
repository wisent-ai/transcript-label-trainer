//! Oko lifecycle-model dataset curation and independent semantic audits.
//!
//! Input rows are masked Oko training envelopes. Brama owns every model call;
//! this module replaces historical silver answers with reviewed decisions and
//! records the exact route used for provenance.

use std::collections::HashSet;
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::brama::{BramaClient, Message};
use crate::util::{now_iso, Error, Result};

mod system_prompt;
mod review_dataset;
mod audit_predictions;

pub use system_prompt::*;
pub use review_dataset::*;
pub use audit_predictions::*;
