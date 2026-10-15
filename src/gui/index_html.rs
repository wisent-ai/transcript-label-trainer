use super::*;

pub(crate) const INDEX_HTML: &[u8] = include_bytes!("../../web/index.html");

pub(crate) const APP_JS: &[u8] = include_bytes!("../../web/app.js");

pub(crate) const STYLES_CSS: &[u8] = include_bytes!("../../web/styles.css");

pub(crate) const DOCS_INDEX: &[u8] = include_bytes!("../../docs/index.html");

pub(crate) const CORPUS_DOCS: &[u8] = include_bytes!("../../docs/corpus-import.html");

pub fn serve(bind: &str, port: i64) -> Result<i32> {
    let ip: IpAddr = bind.parse().map_err(|error| {
        Error(format!(
            "gui --bind must be a loopback IP address, got '{bind}': {error}"
        ))
    })?;
    if !ip.is_loopback() {
        return Err(Error(format!(
            "gui refuses non-loopback bind address {ip}; use the IPv4 or IPv6 loopback address"
        )));
    }
    let port = u16::try_from(port).map_err(|error| {
        Error(format!(
            "gui --port must be a TCP port number, got {port}: {error}"
        ))
    })?;
    let server = tiny_http::Server::http(SocketAddr::new(ip, port))
        .map_err(|error| Error(format!("could not start corpus GUI: {error}")))?;
    let address = server
        .server_addr()
        .to_ip()
        .ok_or_else(|| Error("corpus GUI did not bind an IP socket".to_string()))?;
    let authority = authority(address);
    let origin = format!("http://{authority}");
    // The session token is as wide as a random u128, past the 64 bits of
    // entropy a session identifier needs:
    // https://owasp.org/www-community/vulnerabilities/Insufficient_Session-ID_Length
    let token = hex::encode(rand::random::<u128>().to_be_bytes());

    println!("Corpus importer: {origin}/#token={token}");
    println!("Documentation: {origin}/docs/");
    println!("Press Ctrl-C to stop. The browser is not opened automatically.");

    for request in server.incoming_requests() {
        if let Err(error) = handle(request, &authority, &origin, &token) {
            eprintln!("gui: {error}");
        }
    }
    Ok(crate::cli::EXIT_SUCCESS)
}

/// The upload's outcome counted under why nothing was adopted: a retained
/// bundle of the same identity with other bytes conflicts, anything else
/// rejects. Nothing is imported or unchanged, so those counts are not given.
pub(crate) fn refusal_report(refusal: &corpus::AdoptRefusal) -> Value {
    let conflicting = matches!(refusal, corpus::AdoptRefusal::IdentityConflict(_));
    json!({
        "conflicting": usize::from(conflicting),
        "rejected": usize::from(!conflicting),
    })
}

/// An upload refused before adoption: the refusal's sentence and its count.
fn refused_upload(request: Request, error: String) -> Result<()> {
    let refusal = corpus::AdoptRefusal::Refused(Error(error.clone()));
    respond_json(
        request,
        Status::BAD_REQUEST,
        &json!({"ok": false, "error": error, "report": refusal_report(&refusal)}),
    )
}

/// The retained corpus state, or why it could not be read, beside a refusal.
fn retained_state() -> (Value, Value) {
    match corpus::status() {
        Ok(retained) => (retained, Value::Null),
        Err(error) => (Value::Null, Value::String(error.to_string())),
    }
}

pub(crate) fn upload(mut request: Request, origin: &str, token: &str) -> Result<()> {
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
    if !header(&request, "Content-Type")
        .is_some_and(|value| value.eq_ignore_ascii_case("application/json"))
    {
        return respond_json(
            request,
            Status::UNSUPPORTED_MEDIA_TYPE,
            &json!({"ok": false, "error": "corpus upload Content-Type must be application/json"}),
        );
    }
    let Some(length) = request.body_length() else {
        return respond_json(
            request,
            Status::LENGTH_REQUIRED,
            &json!({"ok": false, "error": "corpus upload requires Content-Length"}),
        );
    };
    let Some(encoded_name) = header(&request, "X-TLT-Filename").map(str::to_string) else {
        return refused_upload(request, "corpus upload requires X-TLT-Filename".to_string());
    };
    let file_name = match percent_decode(&encoded_name) {
        Ok(value) => value,
        Err(error) => return refused_upload(request, error.to_string()),
    };
    let mut raw = Vec::with_capacity(length);
    request.as_reader().read_to_end(&mut raw)?;
    if raw.len() != length {
        return refused_upload(
            request,
            format!(
                "corpus upload carried {} bytes but declared Content-Length {length}",
                raw.len()
            ),
        );
    }
    if raw.is_empty() {
        return refused_upload(request, "corpus upload is empty".to_string());
    }
    match corpus::adopt_upload(&file_name, &raw) {
        Ok(report) => {
            let retained = corpus::status()?;
            respond_json(
                request,
                Status::OK,
                &json!({"ok": true, "report": report, "corpus": retained}),
            )
        }
        Err(refusal) => {
            let (retained, unreadable) = retained_state();
            let report = refusal_report(&refusal);
            respond_json(
                request,
                Status::UNPROCESSABLE_ENTITY,
                &json!({
                    "ok": false,
                    "error": Error::from(refusal).to_string(),
                    "report": report,
                    "corpus": retained,
                    "corpus_error": unreadable,
                }),
            )
        }
    }
}

pub(crate) fn handle(request: Request, authority: &str, origin: &str, token: &str) -> Result<()> {
    if request
        .remote_addr()
        .is_none_or(|address| !address.ip().is_loopback())
    {
        return respond_json(
            request,
            Status::FORBIDDEN,
            &json!({"ok": false, "error": "the GUI accepts loopback clients only"}),
        );
    }
    if header(&request, "Host") != Some(authority) {
        return respond_json(
            request,
            Status::MISDIRECTED_REQUEST,
            &json!({"ok": false, "error": "request Host does not match the GUI listener"}),
        );
    }

    let method = request.method().clone();
    let path = match request.url().split_once('?') {
        Some((path, _query)) => path.to_string(),
        None => request.url().to_string(),
    };
    match (&method, path.as_str()) {
        (&Method::Get, "/") | (&Method::Get, "/index.html") => {
            respond_static(request, "text/html; charset=utf-8", INDEX_HTML)
        }
        (&Method::Get, "/web/app.js") => {
            respond_static(request, "text/javascript; charset=utf-8", APP_JS)
        }
        (&Method::Get, "/web/styles.css") => {
            respond_static(request, "text/css; charset=utf-8", STYLES_CSS)
        }
        (&Method::Get, "/docs") => redirect(request, "/docs/"),
        (&Method::Get, "/docs/") | (&Method::Get, "/docs/index.html") => {
            respond_static(request, "text/html; charset=utf-8", DOCS_INDEX)
        }
        (&Method::Get, "/docs/corpus-import.html") => {
            respond_static(request, "text/html; charset=utf-8", CORPUS_DOCS)
        }
        (&Method::Get, "/api/state") => {
            if !authorized(&request, token) {
                return unauthorized(request);
            }
            match state() {
                Ok(state) => respond_json(request, Status::OK, &state),
                Err(error) => respond_json(
                    request,
                    Status::INTERNAL_SERVER_ERROR,
                    &json!({"ok": false, "error": error.to_string()}),
                ),
            }
        }
        (&Method::Post, "/api/corpora") => upload(request, origin, token),
        (&Method::Post, "/api/train") => train(request, origin, token),
        // corpus-select and corpus-remove: POST /api/corpora/<id>/select, DELETE /api/corpora/<id>.
        (&Method::Post, route) | (&Method::Delete, route) if route.starts_with("/api/corpora/") => {
            corpus_change(request, route, method == Method::Post, origin, token)
        }
        _ => respond_json(
            request,
            Status::NOT_FOUND,
            &json!({"ok": false, "error": "not found"}),
        ),
    }
}

pub(crate) fn state() -> Result<Value> {
    Ok(json!({
        "ok": true,
        "placement": placement::as_dict(),
        "corpus": corpus::status()?,
        "training_keys": {
            (crate::jobs::SKLEARN_MODEL): crate::jobs::TFIDF_KEYS,
            "huggingface": crate::jobs::HF_KEYS,
        },
        "training_presets": crate::jobs::presets_json()?,
    }))
}

pub(crate) fn authority(address: SocketAddr) -> String {
    match address.ip() {
        IpAddr::V4(ip) => format!("{ip}:{}", address.port()),
        IpAddr::V6(ip) => format!("[{ip}]:{}", address.port()),
    }
}

pub(crate) fn authorized(request: &Request, token: &str) -> bool {
    header(request, "X-TLT-Token") == Some(token)
}

pub(crate) fn header<'a>(request: &'a Request, name: &'static str) -> Option<&'a str> {
    request
        .headers()
        .iter()
        .find(|header| header.field.equiv(name))
        .map(|header| header.value.as_str())
}

/// RFC 3986 percent-decoding of a header value: each `%` and the two hex
/// digits after it are one byte, every other byte is itself, and the
/// result must be UTF-8.
pub(crate) fn percent_decode(value: &str) -> Result<String> {
    let invalid = |why: &str| {
        Error(format!(
            "X-TLT-Filename has invalid percent encoding: {why}"
        ))
    };
    let mut decoded = Vec::new();
    let mut bytes = value.bytes();
    while let Some(byte) = bytes.next() {
        if byte != b'%' {
            decoded.push(byte);
            continue;
        }
        let (Some(high), Some(low)) = (bytes.next(), bytes.next()) else {
            return Err(invalid("a % is not followed by two hex digits"));
        };
        let octet = hex::decode([high, low]).map_err(|error| invalid(&error.to_string()))?;
        decoded.extend(octet);
    }
    String::from_utf8(decoded)
        .map_err(|error| Error(format!("X-TLT-Filename is not valid UTF-8: {error}")))
}
