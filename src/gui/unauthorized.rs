use super::*;

pub(crate) fn unauthorized(request: Request) -> Result<()> {
    respond_json(
        request,
        Status::UNAUTHORIZED,
        &json!({"ok": false, "error": "missing or invalid GUI session token"}),
    )
}

pub(crate) fn redirect(request: Request, location: &str) -> Result<()> {
    let response = Response::empty(StatusCode(Status::PERMANENT_REDIRECT.as_u16()))
        .with_header(header_from("Location", location))
        .with_header(header_from("Cache-Control", "no-store"));
    request
        .respond(response)
        .map_err(|error| Error(format!("could not send GUI response: {error}")))
}

pub(crate) fn respond_json(request: Request, status: Status, value: &Value) -> Result<()> {
    let body = serde_json::to_vec(value)?;
    let length = body.len();
    let response = Response::new(
        StatusCode(status.as_u16()),
        response_headers("application/json; charset=utf-8"),
        Cursor::new(body),
        Some(length),
        None,
    );
    request
        .respond(response)
        .map_err(|error| Error(format!("could not send GUI response: {error}")))
}

pub(crate) fn respond_static(
    request: Request,
    content_type: &str,
    body: &'static [u8],
) -> Result<()> {
    let response = Response::new(
        StatusCode(Status::OK.as_u16()),
        response_headers(content_type),
        Cursor::new(body),
        Some(body.len()),
        None,
    );
    request
        .respond(response)
        .map_err(|error| Error(format!("could not send GUI response: {error}")))
}

pub(crate) fn response_headers(content_type: &str) -> Vec<Header> {
    vec![
        header_from("Content-Type", content_type),
        header_from("Cache-Control", "no-store"),
        header_from("X-Content-Type-Options", "nosniff"),
        header_from("Referrer-Policy", "no-referrer"),
        header_from("X-Frame-Options", "DENY"),
        header_from(
            "Content-Security-Policy",
            "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
        ),
    ]
}

/// A response header from the GUI's own names and values; the only values
/// that do not come from this file are paths it routes, which are ASCII.
pub(crate) fn header_from(name: &str, value: &str) -> Header {
    match Header::from_bytes(name.as_bytes(), value.as_bytes()) {
        Ok(header) => header,
        Err(()) => panic!("the GUI built a response header tiny_http refuses: {name}: {value}"),
    }
}
