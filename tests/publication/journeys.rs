use std::fs;
use std::path::Path;

use serde_json::{json, Value};

use super::support::{blocked, digest, require, Context, Result};

pub fn public_refusal(context: &mut Context) -> Result<Value> {
    context.with_repository(Some(false), |context, repo| {
        let before = context.info(repo, None)?.ok_or("public fixture was not created")?;
        require(before["private"] == false, "fixture is not public")?;
        let receipt = context.work.join("public-refusal.json");
        let missing = context.work.join("absent-training-input");
        let mut command = context.product();
        command.arg("humanizer-publish").arg(&missing)
            .args(["--repo", repo]).arg("--metrics").arg(&missing)
            .arg("--audit").arg(&missing).arg("--preparation").arg(&missing)
            .arg("--output").arg(&receipt);
        let output = context.run("public-refusal", command)?;
        require(output.status.code() == Some(1), format!("public destination returned {}", output.status))?;
        let report: Value = serde_json::from_slice(&fs::read(&receipt)?)?;
        require(report["qualified"] == false && report["error"] == "repository_not_private",
            format!("public destination did not report its actual refusal: {report}"))?;
        require(report["repo_id"] == repo && report["observed_private"] == false,
            "refusal does not identify the observed public destination")?;
        let after = context.info(repo, None)?.ok_or("public destination disappeared")?;
        require(after["private"] == false && after["sha"] == before["sha"]
            && after["siblings"] == before["siblings"], "refusal changed the public destination")?;
        Ok(json!({"repository": repo, "before": before, "after": after, "receipt": report}))
    })
}

pub fn private_publication(context: &mut Context) -> Result<Value> {
    let Some(fixture) = std::env::var_os("TLT_PUBLICATION_FIXTURE") else {
        return blocked("TLT_PUBLICATION_FIXTURE must name a real qualified humanizer job output");
    };
    let fixture = Path::new(&fixture).canonicalize()?;
    let model = fixture.join("adapter");
    let metrics = fixture.join("metrics.json");
    let audit = fixture.join("audit.json");
    let preparation = fixture.join("preparation.json");
    let mut expected = Vec::new();
    for (name, source) in [
        ("adapter_config.json", model.join("adapter_config.json")),
        ("adapter_model.safetensors", model.join("adapter_model.safetensors")),
        ("tokenizer.json", model.join("tokenizer.json")),
        ("evaluation/metrics.json", metrics.clone()),
        ("evaluation/audit.json", audit.clone()),
        ("evaluation/preparation.json", preparation.clone()),
    ] {
        if !source.is_file() {
            return blocked(format!("qualified fixture is missing {}", source.display()));
        }
        expected.push((name, digest(&source)?));
    }
    let verdict: Value = serde_json::from_slice(&fs::read(&audit)?)?;
    if verdict["passed"] != true {
        return blocked(format!("fixture {} has no passed real audit", audit.display()));
    }
    context.with_repository(None, |context, repo| {
        let receipt = context.work.join("private-publication.json");
        let mut command = context.product();
        command.arg("humanizer-publish").arg(&model).args(["--repo", repo])
            .arg("--metrics").arg(&metrics).arg("--audit").arg(&audit)
            .arg("--preparation").arg(&preparation).arg("--output").arg(&receipt);
        let output = context.run("private-publication", command)?;
        require(output.status.success(), format!("private publication returned {}", output.status))?;
        let report: Value = serde_json::from_slice(&fs::read(&receipt)?)?;
        require(report["qualified"] == true && report["private"] == true && report["repo_id"] == repo,
            format!("publication returned an invalid receipt: {report}"))?;
        let revision = report["revision"].as_str().ok_or("receipt has no immutable revision")?;
        let observed = context.info(repo, Some(revision))?.ok_or("published revision is absent")?;
        require(observed["private"] == true && observed["sha"] == revision,
            "provider does not confirm the receipt's private immutable revision")?;
        let downloads = context.work.join("downloaded");
        let mut verified = Vec::new();
        for (name, hash) in &expected {
            let mut command = context.hf();
            command.args(["download", repo, name, "--revision", revision, "--local-dir"])
                .arg(&downloads).arg("--quiet");
            let output = context.run("download", command)?;
            require(output.status.success(), format!("download {name} returned {}", output.status))?;
            let actual = digest(&downloads.join(name))?;
            require(&actual == hash, format!("downloaded {name}: expected {hash}, observed {actual}"))?;
            require(report["verified_files"][name]["sha256"] == *hash,
                format!("publication receipt does not bind {name} to its actual bytes"))?;
            verified.push(json!({"path": name, "sha256": actual}));
        }
        Ok(json!({"repository": repo, "receipt": report, "observed": observed, "downloaded": verified}))
    })
}
