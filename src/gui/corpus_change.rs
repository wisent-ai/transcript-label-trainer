//! The window's `corpus-select` and `corpus-remove`: POST
//! /api/corpora/<id>/select and DELETE /api/corpora/<id>, guarded like the
//! upload (session token and the listener's exact Origin) and answered with
//! the registry after the change, or the refusal sentence.

use super::*;

pub(crate) fn corpus_change(
    request: Request,
    route: &str,
    post: bool,
    origin: &str,
    token: &str,
) -> Result<()> {
    if !authorized(&request, token) {
        return unauthorized(request);
    }
    if header(&request, "Origin") != Some(origin) {
        return respond_json(
            request,
            Status::FORBIDDEN,
            &json!({"ok": false, "error": "mutation Origin does not match the GUI listener"}),
        );
    }
    let Some(rest) = route.strip_prefix("/api/corpora/") else {
        return respond_json(
            request,
            Status::NOT_FOUND,
            &json!({"ok": false, "error": "not found"}),
        );
    };
    let (encoded, select) = match (post, rest.strip_suffix("/select")) {
        (true, Some(id)) => (id, true),
        (false, None) => (rest, false),
        _ => {
            return respond_json(
                request,
                Status::NOT_FOUND,
                &json!({"ok": false, "error": "not found"}),
            )
        }
    };
    let outcome = match percent_decode(encoded) {
        Ok(id) if select => corpus::select(&id),
        Ok(id) => corpus::remove(&id),
        Err(error) => Err(error),
    };
    match outcome {
        Ok(retained) => respond_json(
            request,
            Status::OK,
            &json!({"ok": true, "corpus": retained}),
        ),
        Err(error) => respond_json(
            request,
            Status::UNPROCESSABLE_ENTITY,
            &json!({"ok": false, "error": error.to_string()}),
        ),
    }
}
