//! Training, inference, and artifact inspection for aspect-label classifiers.
//!
//! Two backends share one data path (lake label store + lake CLI session text):
//!
//! - tfidf-logreg (default): TF-IDF + multinomial logistic regression,
//!   artifacts directly in `<training root>/models/<aspect>/` (`model.json` +
//!   `metrics.json`);
//! - hf (`--model <hf-model-id>`, optional `hf` feature): fine-tuned
//!   sequence classifier in `<training root>/models/<aspect>/hf-<id>/`.
//!
//! The training root is resolved by [`crate::placement`], never read straight
//! out of the environment here.
//!
//! Neither backend ever trains on the frozen evaluation split:
//! [`crate::evaluate::resolve_split`] resolves it before training starts, both
//! backends fit on the training side only, and both report the holdout under
//! `holdout_evaluation`.
//!
//! When both backends exist for an aspect, inference uses the newest artifact
//! by `trained_at`.
//!
//! The Python original delegated the vectorizer and the classifier to sklearn
//! and froze the fitted pipeline into `model.joblib`. A pickle of Python
//! objects is not portable to this binary, so the artifact is `model.json`:
//! the vocabulary, the idf vector, the class list and the weight matrix, in
//! the one format anything can read. Everything around it — `metrics.json`,
//! `eval-split.json`, `job.yaml`, the metric keys, the guard rails — is
//! unchanged, because `info`, `evaluate` and the docs read those.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use crate::util::{Error, Result, TrainFailure};
use crate::{evaluate, jobs, lake, placement};

mod min_labeled_sessions_group;
mod train_tfidf_group;

pub use min_labeled_sessions_group::*;
pub use train_tfidf_group::*;
