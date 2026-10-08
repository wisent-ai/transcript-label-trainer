//! How many calls at once a command fans out to one Brama route: the
//! gateway's own measurement, read from `GET /v1/aliases`'s
//! `measured_concurrency`, never a count stated on the command line.
//!
//! `carried` is the most calls in flight the route has answered at once. With
//! no capacity refusal at that level the command asks for one call more, so
//! the measurement can grow run by run; when the route was refused for
//! capacity with `carried` or fewer calls beside it, the command stays at
//! `carried`. A route the gateway has measured nothing for yet starts at one
//! call at a time, as does a route refused with nothing else in flight.

use std::num::NonZeroUsize;

use serde::Deserialize;

use super::BramaClient;
use crate::bail;
use crate::util::Result;

#[derive(Deserialize)]
struct Measured {
    carried: Option<usize>,
    refused_beside: Option<usize>,
}

#[derive(Deserialize)]
struct Listing {
    #[serde(default)]
    measured_concurrency: std::collections::BTreeMap<String, Measured>,
}

/// The allowance a measurement grants.
fn granted(measured: Option<&Measured>) -> NonZeroUsize {
    let Some(carried) = measured.and_then(|measured| measured.carried) else {
        return NonZeroUsize::MIN;
    };
    let refused_at_or_below = measured
        .and_then(|measured| measured.refused_beside)
        .is_some_and(|beside| beside <= carried);
    match (refused_at_or_below, NonZeroUsize::new(carried)) {
        (false, _) => NonZeroUsize::MIN.saturating_add(carried),
        (true, Some(carried)) => carried,
        (true, None) => NonZeroUsize::MIN,
    }
}

impl BramaClient {
    /// The calls at once this command may send to `model`.
    pub fn allowance(&self, model: &str) -> Result<NonZeroUsize> {
        let mut request = self.http.get(format!("{}/v1/aliases", self.url));
        for (name, value) in self.auth_headers("") {
            request = request.header(name, value);
        }
        let response = match request.send() {
            Ok(response) => response,
            Err(error) => bail!("Brama unreachable at {} reading its measured concurrency: {error}", self.url),
        };
        let status = response.status();
        let text = match response.text() {
            Ok(text) => text,
            Err(error) => bail!("Brama's GET /v1/aliases body could not be read: {error}"),
        };
        if !status.is_success() {
            bail!("Brama answered GET /v1/aliases with HTTP {}: {}", status.as_u16(), text.trim())
        }
        let listing: Listing = match serde_json::from_str(&text) {
            Ok(listing) => listing,
            Err(error) => bail!("Brama's GET /v1/aliases is not the alias listing ({error}): {}", text.trim()),
        };
        Ok(granted(listing.measured_concurrency.get(model)))
    }
}

/// The allowance for `model`, read with the environment's Brama identity.
pub fn allowance(model: &str) -> Result<NonZeroUsize> {
    BramaClient::from_env()?.allowance(model)
}
