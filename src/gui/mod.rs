//! Loopback-only browser workspace for adopting an existing corpus.
//!
//! The frontend is compiled into the binary. Uploads stay in memory until the
//! corpus module has validated them; that module is also the implementation
//! behind `corpus-adopt`, so the GUI cannot drift into a second import path.
use std::io::{Cursor, Read};
use std::net::{IpAddr, SocketAddr};

use serde_json::{json, Value};
use tiny_http::{Header, Method, Request, Response, StatusCode};

use crate::util::{Error, Result};
use crate::{corpus, placement};

mod index_html;
mod unauthorized;

pub use index_html::*;
pub use unauthorized::*;
