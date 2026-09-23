//! The HuggingFace fine-tune backend, ported from `model.py`'s `_train_hf`,
//! `_hf_predict` and `_infer_hf` onto candle. Compiled only under the `hf`
//! feature; when it is off, `model.rs` refuses `--model` with the actionable
//! rebuild message and never reaches this file.
//!
//! What the Python path did with torch + transformers this does with candle:
//! pull the base model from the hub, put a sequence-classification head on it,
//! fine-tune the whole thing with AdamW, and save weights, tokenizer and the
//! metrics fragment into `<aspect dir>/hf-<sanitized model id>/`.
//!
//! Two architectures load, the two an operator training mixed Polish/English
//! transcripts would reach for: `distilbert` (the documented default,
//! `distilbert-base-multilingual-cased`) and `bert`. Anything else fails with
//! a sentence naming what is supported instead of loading garbage weights.
//!
//! The encoders are spelled out here rather than taken from
//! candle-transformers, and the reason is load-bearing: that crate's blocks
//! normalise through `candle_nn::LayerNorm`, which dispatches to a fused
//! kernel registered with `apply_op3_no_bwd`. A tensor produced by that kernel
//! carries no backward op, so `loss.backward()` stops at the first layer norm
//! and the gradient never reaches the encoder. Measured against
//! `distilbert-base-multilingual-cased`: 4 of 104 parameters received a
//! gradient, and every encoder weight came out of a fine-tune bit-identical to
//! the checkpoint. The same blocks written out of primitive ops (including
//! `ops::layer_norm_slow`) differentiate end to end.
//!
//! `model.rs` owns everything around this: the base metrics, the frozen
//! evaluation split, `holdout_evaluation`, the job block, and writing
//! `metrics.json` into the directory `train` returns.
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, IndexOp, Tensor, Var, D};
use candle_nn::{loss, ops, AdamW, Embedding, Linear, Module, Optimizer, ParamsAdamW, VarBuilder};
use hf_hub::api::sync::Api;
use rand::seq::SliceRandom;
use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::Deserialize;
use serde_json::{json, Map, Value};
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

use crate::util::{Error, Result, TrainFailure};

mod min_per_class_group;
mod hub_files_group;

pub use min_per_class_group::*;
pub use hub_files_group::*;
