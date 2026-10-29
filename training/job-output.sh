# Sourced by the Stado job scripts under training/: where a job stages what
# Stado uploads to its --output-uri.
#
# Stado starts a job's command in that job's own directory, wc-<job id> under
# the host's work root, checks the repository out into the --repo-workdir
# child of it, and uploads the directory's output/ when the job ends
# (stado-rs providers/local/slots/lifecycle: start.rs current_dir, advance.rs
# work_dir.join("output")). The output directory is therefore the sibling of
# this checkout, read from where the job runs, never a path guessed from an
# older agent's layout.

# job_output_dir CHECKOUT: print the output directory of the Stado job that
# checked CHECKOUT out, or refuse naming what CHECKOUT sits in.
job_output_dir() {
  local checkout="$1" job_dir
  job_dir="$(dirname "$checkout")"
  if [[ "$(basename "$job_dir")" == "wc-$WC_JOB_ID" ]]; then
    printf '%s\n' "$job_dir/output"
  else
    echo "job output: the checkout $checkout sits in $job_dir, not in the directory wc-$WC_JOB_ID Stado runs job $WC_JOB_ID in; run this script as a Stado job submitted with --repo-workdir" >&2
    false
  fi
}
