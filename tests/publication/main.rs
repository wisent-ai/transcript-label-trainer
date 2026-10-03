use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::{json, Value};

mod journeys;
mod support;
use support::{blocked, digest, Blocked, Context, Result, BINARY};

fn source(root: &Path, report: &mut Value) -> Result<()> {
    let revision = Command::new("git").args(["rev-parse", "HEAD"]).current_dir(root).output()?;
    support::require(revision.status.success(), String::from_utf8_lossy(&revision.stderr))?;
    report["source_revision"] = json!(String::from_utf8(revision.stdout)?.trim());
    let state = Command::new("git").args(["status", "--porcelain", "--untracked-files=normal"])
        .current_dir(root).output()?;
    support::require(state.status.success(), String::from_utf8_lossy(&state.stderr))?;
    report["source_state"] = json!(String::from_utf8(state.stdout)?);
    report["binary_sha256"] = json!(digest(Path::new(BINARY))?);
    if report["source_state"] != "" {
        return blocked("real publication qualification requires a clean committed source checkout");
    }
    Ok(())
}

fn run(root: &Path, work: &Path, report: &mut Value) -> Result<i32> {
    source(root, report)?;
    let token = match std::env::var("HF_TOKEN") {
        Ok(token) if !token.trim().is_empty() => token,
        _ => return blocked("HF_TOKEN is required for isolated real Hugging Face repositories"),
    };
    let mut context = Context::new(work.to_path_buf(), token)?;
    let outcome = (|| -> Result<i32> {
        let identity = context.get("/api/whoami-v2")?.ok_or("Hugging Face identity endpoint is unavailable")?;
        context.owner = match std::env::var("TLT_PUBLICATION_NAMESPACE") {
            Ok(value) => value,
            Err(std::env::VarError::NotPresent) => identity["name"].as_str()
                .ok_or("Hugging Face identity contains no account name")?.to_owned(),
            Err(error) => return blocked(format!("cannot read TLT_PUBLICATION_NAMESPACE: {error}")),
        };
        support::require(!context.owner.is_empty() && context.owner.chars()
            .all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c)), "invalid fixture namespace")?;
        let mut exit = 0;
        let cases: [(&str, fn(&mut Context) -> Result<Value>); 2] = [
            ("public_refusal", journeys::public_refusal),
            ("private_publication", journeys::private_publication),
        ];
        let mut results = Vec::new();
        for (name, journey) in cases {
            let result = match journey(&mut context) {
                Ok(evidence) => json!({"name": name, "status": "passed", "evidence": evidence}),
                Err(error) => {
                    let blocked = error.is::<Blocked>();
                    if !blocked { exit = 1; } else if exit == 0 { exit = 2; }
                    json!({"name": name, "status": if blocked { "blocked" } else { "failed" },
                        "error": error.to_string()})
                }
            };
            results.push(result);
            report["cases"] = json!(results);
            fs::write(work.join("report.json"), serde_json::to_vec_pretty(report)?)?;
        }
        Ok(exit)
    })();
    report["commands"] = json!(context.commands);
    report["http_requests"] = json!(context.requests);
    outcome
}

fn execute() -> Result<i32> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).canonicalize()?;
    let build = root.join(".build");
    support::require(!build.is_symlink(), "refusing a symlinked build directory")?;
    fs::create_dir_all(&build)?;
    let work = build.join(format!("publication-{}", hex::encode(rand::random::<[u8; 16]>())));
    let mut directory = fs::DirBuilder::new();
    #[cfg(unix)] {
        use std::os::unix::fs::DirBuilderExt;
        directory.mode(0o700);
    }
    directory.create(&work)?;
    let mut report = json!({"schema_version": 1, "started_at": chrono::Utc::now().to_rfc3339(), "cases": []});
    let mut exit = match run(&root, &work, &mut report) {
        Ok(exit) => exit,
        Err(error) => {
            report["error"] = json!(error.to_string());
            if error.is::<Blocked>() { 2 } else { 1 }
        }
    };
    for directory in ["downloaded", "hf-home"] {
        let path = work.join(directory);
        if path.exists() {
            if let Err(error) = fs::remove_dir_all(&path) {
                report["local_cleanup_error"] = json!(format!("{}: {error}", path.display()));
                exit = 1;
            }
        }
    }
    report["status"] = json!(match exit { 0 => "passed", 2 => "blocked", _ => "failed" });
    report["finished_at"] = json!(chrono::Utc::now().to_rfc3339());
    fs::write(work.join("report.json"), serde_json::to_vec_pretty(&report)?)?;
    println!("{}", json!({"report": work.join("report.json"), "status": report["status"]}));
    Ok(exit)
}

fn main() {
    let exit = match execute() {
        Ok(exit) => exit,
        Err(error) => { eprintln!("publication qualification could not retain its result: {error}"); 1 }
    };
    std::process::exit(exit);
}
