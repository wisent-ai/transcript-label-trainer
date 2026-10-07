//! Loopback-only browser workspace for adopting an existing corpus and
//! training a classifier from it.
//!
//! The frontend is compiled into the binary. Uploads stay in memory until the
//! corpus module has validated them; that module is also the implementation
//! behind `corpus-adopt`, and training runs through `model::train`, the
//! implementation behind `train`, so the GUI cannot drift into a second path.
use std::io::{Cursor, Read};
use std::net::{IpAddr, SocketAddr};

use serde_json::{json, Value};
use tiny_http::{Header, Method, Request, Response, StatusCode};

use crate::util::{Error, Result};
use crate::{corpus, placement};

mod corpus_change;
mod index_html;
mod train;
mod unauthorized;

pub use corpus_change::*;
pub use index_html::*;
pub use train::*;
pub use unauthorized::*;
