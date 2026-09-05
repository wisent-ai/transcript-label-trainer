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

const INDEX_HTML: &[u8] = include_bytes!("../web/index.html");
const APP_JS: &[u8] = include_bytes!("../web/app.js");
const STYLES_CSS: &[u8] = include_bytes!("../web/styles.css");
const DOCS_INDEX: &[u8] = include_bytes!("../docs/index.html");
const CORPUS_DOCS: &[u8] = include_bytes!("../docs/corpus-import.html");
const MAX_UPLOAD_BYTES: usize = 16 * 1024 * 1024;

pub fn serve(bind: &str, port: i64) -> Result<i32> {
    let ip: IpAddr = bind
        .parse()
        .map_err(|_| Error(format!("gui --bind must be a loopback IP address, got '{bind}'")))?;
    if !ip.is_loopback() {
        return Err(Error(format!(
            "gui refuses non-loopback bind address {ip}; use 127.0.0.1 or ::1"
        )));
    }
    let port = u16::try_from(port)
        .map_err(|_| Error(format!("gui --port must be between 0 and 65535, got {port}")))?;
    let server = tiny_http::Server::http(SocketAddr::new(ip, port))
        .map_err(|error| Error(format!("could not start corpus GUI: {error}")))?;
    let address = server
        .server_addr()
        .to_ip()
        .ok_or_else(|| Error("corpus GUI did not bind an IP socket".to_string()))?;
    let authority = authority(address);
    let origin = format!("http://{authority}");
    let token = hex::encode(rand::random::<[u8; 32]>());

    println!("Corpus importer: {origin}/#token={token}");
    println!("Documentation: {origin}/docs/");
    println!("Press Ctrl-C to stop. The browser is not opened automatically.");

    for request in server.incoming_requests() {
        if let Err(error) = handle(request, &authority, &origin, &token) {
            eprintln!("gui: {error}");
        }
    }
    Ok(0)
}

fn handle(mut request: Request, authority: &str, origin: &str, token: &str) -> Result<()> {
    if request
        .remote_addr()
        .is_none_or(|address| !address.ip().is_loopback())
    {
        return respond_json(
            request,
            403,
            &json!({"ok": false, "error": "the GUI accepts loopback clients only"}),
        );
    }
    if header(&request, "Host") != Some(authority) {
        return respond_json(
            request,
            421,
            &json!({"ok": false, "error": "request Host does not match the GUI listener"}),
        );
    }

    let method = request.method().clone();
    let path = request
        .url()
        .split('?')
        .next()
        .unwrap_or(request.url())
        .to_string();
    match (&method, path.as_str()) {
        (&Method::Get, "/") | (&Method::Get, "/index.html") => {
            respond_static(request, 200, "text/html; charset=utf-8", INDEX_HTML)
        }
        (&Method::Get, "/web/app.js") => {
            respond_static(request, 200, "text/javascript; charset=utf-8", APP_JS)
        }
        (&Method::Get, "/web/styles.css") => {
            respond_static(request, 200, "text/css; charset=utf-8", STYLES_CSS)
        }
        (&Method::Get, "/docs") => redirect(request, "/docs/"),
        (&Method::Get, "/docs/") | (&Method::Get, "/docs/index.html") => {
            respond_static(request, 200, "text/html; charset=utf-8", DOCS_INDEX)
        }
        (&Method::Get, "/docs/corpus-import.html") => {
            respond_static(request, 200, "text/html; charset=utf-8", CORPUS_DOCS)
        }
        (&Method::Get, "/api/state") => {
            if !authorized(&request, token) {
                return unauthorized(request);
            }
            match state() {
                Ok(state) => respond_json(request, 200, &state),
                Err(error) => respond_json(
                    request,
                    500,
                    &json!({"ok": false, "error": error.to_string()}),
                ),
            }
        }
        (&Method::Post, "/api/corpora") => {
            if !authorized(&request, token) {
                return unauthorized(request);
            }
            if header(&request, "Origin") != Some(origin) {
                return respond_json(
                    request,
                    403,
                    &json!({"ok": false, "error": "mutation Origin does not match the GUI listener"}),
                );
            }
            if !header(&request, "Content-Type")
                .is_some_and(|value| value.eq_ignore_ascii_case("application/json"))
            {
                return respond_json(
                    request,
                    415,
                    &json!({"ok": false, "error": "corpus upload Content-Type must be application/json"}),
                );
            }
            let Some(length) = request.body_length() else {
                return respond_json(
                    request,
                    411,
                    &json!({"ok": false, "error": "corpus upload requires Content-Length"}),
                );
            };
            if length == 0 || length > MAX_UPLOAD_BYTES {
                return respond_json(
                    request,
                    413,
                    &json!({
                        "ok": false,
                        "error": format!("corpus upload must contain 1-{MAX_UPLOAD_BYTES} bytes"),
                        "report": empty_report(1, 0),
                    }),
                );
            }
            let encoded_name = match header(&request, "X-TLT-Filename") {
                Some(value) => value.to_string(),
                None => {
                    return respond_json(
                        request,
                        400,
                        &json!({"ok": false, "error": "corpus upload requires X-TLT-Filename"}),
                    )
                }
            };
            let file_name = match percent_decode(&encoded_name) {
                Ok(value) => value,
                Err(error) => {
                    return respond_json(
                        request,
                        400,
                        &json!({"ok": false, "error": error.to_string()}),
                    )
                }
            };
            let mut raw = Vec::with_capacity(length);
            request
                .as_reader()
                .take((MAX_UPLOAD_BYTES + 1) as u64)
                .read_to_end(&mut raw)?;
            if raw.len() != length || raw.len() > MAX_UPLOAD_BYTES {
                return respond_json(
                    request,
                    413,
                    &json!({
                        "ok": false,
                        "error": "corpus upload length did not match the bounded request",
                        "report": empty_report(1, 0),
                    }),
                );
            }
            match corpus::adopt_upload(&file_name, &raw) {
                Ok(report) => {
                    let retained = corpus::status()?;
                    respond_json(
                        request,
                        200,
                        &json!({"ok": true, "report": report, "corpus": retained}),
                    )
                }
                Err(error) => {
                    let conflict = usize::from(error.0.contains("corpus identity conflict"));
                    let retained = corpus::status().ok();
                    respond_json(
                        request,
                        422,
                        &json!({
                            "ok": false,
                            "error": error.to_string(),
                            "report": empty_report(1 - conflict, conflict),
                            "corpus": retained,
                        }),
                    )
                }
            }
        }
        _ => respond_json(
            request,
            404,
            &json!({"ok": false, "error": "not found"}),
        ),
    }
}

fn state() -> Result<Value> {
    Ok(json!({
        "ok": true,
        "placement": placement::as_dict(),
        "corpus": corpus::status()?,
        "maxUploadBytes": MAX_UPLOAD_BYTES,
    }))
}

fn empty_report(rejected: usize, conflicting: usize) -> Value {
    json!({
        "imported": 0,
        "unchanged": 0,
        "conflicting": conflicting,
        "rejected": rejected,
    })
}

fn authority(address: SocketAddr) -> String {
    match address.ip() {
        IpAddr::V4(ip) => format!("{ip}:{}", address.port()),
        IpAddr::V6(ip) => format!("[{ip}]:{}", address.port()),
    }
}

fn authorized(request: &Request, token: &str) -> bool {
    header(request, "X-TLT-Token") == Some(token)
}

fn header<'a>(request: &'a Request, name: &'static str) -> Option<&'a str> {
    request
        .headers()
        .iter()
        .find(|header| header.field.equiv(name))
        .map(|header| header.value.as_str())
}

fn percent_decode(value: &str) -> Result<String> {
    let source = value.as_bytes();
    let mut decoded = Vec::with_capacity(source.len());
    let mut index = 0;
    while index < source.len() {
        if source[index] == b'%' {
            if index + 2 >= source.len() {
                return Err(Error("X-TLT-Filename has invalid percent encoding".to_string()));
            }
            let high = hex_digit(source[index + 1])?;
            let low = hex_digit(source[index + 2])?;
            decoded.push((high << 4) | low);
            index += 3;
        } else {
            decoded.push(source[index]);
            index += 1;
        }
    }
    String::from_utf8(decoded)
        .map_err(|_| Error("X-TLT-Filename is not valid UTF-8".to_string()))
}

fn hex_digit(byte: u8) -> Result<u8> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        b'A'..=b'F' => Ok(byte - b'A' + 10),
        _ => Err(Error(
            "X-TLT-Filename has invalid percent encoding".to_string(),
        )),
    }
}

fn unauthorized(request: Request) -> Result<()> {
    respond_json(
        request,
        401,
        &json!({"ok": false, "error": "missing or invalid GUI session token"}),
    )
}

fn redirect(request: Request, location: &str) -> Result<()> {
    let response = Response::empty(StatusCode(308))
        .with_header(header_from("Location", location))
        .with_header(header_from("Cache-Control", "no-store"));
    request
        .respond(response)
        .map_err(|error| Error(format!("could not send GUI response: {error}")))
}

fn respond_json(request: Request, status: u16, value: &Value) -> Result<()> {
    let body = serde_json::to_vec(value)?;
    let length = body.len();
    let response = Response::new(
        StatusCode(status),
        response_headers("application/json; charset=utf-8"),
        Cursor::new(body),
        Some(length),
        None,
    );
    request
        .respond(response)
        .map_err(|error| Error(format!("could not send GUI response: {error}")))
}

fn respond_static(
    request: Request,
    status: u16,
    content_type: &str,
    body: &'static [u8],
) -> Result<()> {
    let response = Response::new(
        StatusCode(status),
        response_headers(content_type),
        Cursor::new(body),
        Some(body.len()),
        None,
    );
    request
        .respond(response)
        .map_err(|error| Error(format!("could not send GUI response: {error}")))
}

fn response_headers(content_type: &str) -> Vec<Header> {
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

fn header_from(name: &str, value: &str) -> Header {
    Header::from_bytes(name.as_bytes(), value.as_bytes())
        .expect("static and validated response header")
}
