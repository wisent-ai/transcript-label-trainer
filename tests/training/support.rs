use std::error::Error;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

pub type Result<T, E = Box<dyn Error>> = std::result::Result<T, E>;
pub const BINARY: &str = env!("CARGO_BIN_EXE_transcript-label-trainer");

pub fn required(name: &str) -> Result<String> {
    let value = std::env::var(name).map_err(|error| format!("{name}: {error}"))?;
    require(!value.trim().is_empty(), format!("{name} must be stated"))?;
    Ok(value)
}

pub fn require(condition: bool, message: impl Into<String>) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(message.into().into())
    }
}

pub fn digest(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    std::io::copy(&mut file, &mut hash)?;
    Ok(hex::encode(hash.finalize()))
}

pub struct Evidence {
    pub root: PathBuf,
    pub report: Value,
}

impl Evidence {
    pub fn save(&self) -> Result<()> {
        fs::write(
            self.root.join("report.json"),
            serde_json::to_vec_pretty(&self.report)?,
        )?;
        Ok(())
    }

    pub fn run(
        &mut self,
        program: &str,
        args: &[&str],
        source: Option<&str>,
    ) -> Result<(bool, String, String)> {
        let ordinal = self.report["commands"]
            .as_array()
            .ok_or("commands are not an array")?
            .len();
        let stdout = self.root.join(format!("{ordinal}.stdout"));
        let stderr = self.root.join(format!("{ordinal}.stderr"));
        let mut command = Command::new(program);
        command
            .args(args)
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .stdout(File::create(&stdout)?)
            .stderr(File::create(&stderr)?);
        if let Some(source) = source {
            command.env("TLT_REPO_REF", source);
        }
        self.report["commands"]
            .as_array_mut()
            .ok_or("commands are not an array")?
            .push(json!({
                "program": program, "args": args, "source": source,
                "stdout": stdout, "stderr": stderr, "started_at": chrono::Utc::now().to_rfc3339(),
            }));
        self.save()?;
        let status = command.status()?;
        self.report["commands"][ordinal]["exit_status"] = json!(status.code());
        self.report["commands"][ordinal]["status"] = json!(status.to_string());
        self.save()?;
        Ok((
            status.success(),
            fs::read_to_string(stdout)?,
            fs::read_to_string(stderr)?,
        ))
    }

    pub fn succeeds(
        &mut self,
        program: &str,
        args: &[&str],
        source: Option<&str>,
    ) -> Result<String> {
        let (success, stdout, stderr) = self.run(program, args, source)?;
        require(success, format!("{program} {args:?}: {stderr}"))?;
        Ok(stdout)
    }
}

pub fn field<'a>(stdout: &'a str, prefix: &str) -> Result<&'a str> {
    stdout
        .lines()
        .find_map(|line| line.strip_prefix(prefix))
        .map(str::trim)
        .ok_or_else(|| format!("missing {prefix:?} in actual product output: {stdout}").into())
}
