use super::*;

pub(crate) fn build_specs() -> Vec<Spec> {
    let mut specs = training_specs();
    specs.extend(corpus_specs());
    specs.extend(model_specs());
    specs
}

#[allow(unused_variables)]
fn training_specs() -> Vec<Spec> {
    let fraction = float_repr(jobs::DEFAULT_EVAL_FRACTION);
    let seed = jobs::DEFAULT_EVAL_SEED;
    let teacher = brama::DEFAULT_MODEL;

    let train = Spec {
        name: "train",
        help: "train a classifier for one aspect".to_string(),
        description: None,
        positionals: Vec::new(),
        opts: vec![
            required(
                "--aspect",
                "ASPECT",
                Kind::Text,
                "aspect name, e.g. reviewed".to_string(),
            ),
            option(
                "--model",
                "HF_MODEL_ID",
                Kind::Text,
                "fine-tune this HuggingFace model instead of the default TF-IDF + \
                 logistic regression (requires the 'hf' extra); multilingual models \
                 such as distilbert-base-multilingual-cased fit the mixed \
                 Polish/English transcripts"
                    .to_string(),
            ),
            option(
                "--epochs",
                "EPOCHS",
                Kind::Float,
                "HF training epochs (default: 3)".to_string(),
            ),
            option(
                "--batch-size",
                "BATCH_SIZE",
                Kind::Int,
                "HF batch size (default: 8)".to_string(),
            ),
            option(
                "--lr",
                "LR",
                Kind::Float,
                "HF learning rate (default: 2e-5)".to_string(),
            ),
            option(
                "--max-length",
                "MAX_LENGTH",
                Kind::Int,
                "HF tokenizer max tokens per session (default: 512)".to_string(),
            ),
            option(
                "--eval-split-fraction",
                "F",
                Kind::Float,
                format!(
                    "share of labeled sessions frozen out of training the first time \
                     this aspect is trained (default: {fraction}); later runs reuse \
                     the frozen eval-split.json unchanged"
                ),
            ),
            option(
                "--eval-split-seed",
                "N",
                Kind::Int,
                format!("seed that picks the frozen holdout (default: {seed})"),
            ),
            option(
                "--no-eval-split",
                "",
                Kind::Flag,
                "train on every labeled session, with no frozen holdout to evaluate on".to_string(),
            ),
        ],
    };

    let run = Spec {
        name: "run",
        help: "execute a declarative training job (YAML spec)".to_string(),
        description: Some(format!(
            "Execute a declarative training job. Two spec sections are on unless the \
             spec turns them off. 'eval_split' (fraction: {fraction}, seed: {seed}) \
             freezes a holdout of labeled sessions into \
             <training root>/models/<name>/eval-split.json the first time the job \
             runs; every later run reuses that file unchanged, trains on nothing in \
             it, and reports it under 'holdout_evaluation' in metrics.json. \
             'eval_split: false' trains on every labeled session. 'judge' (model: \
             {teacher}) names the Brama-routed teacher that 'evaluate' asks for a \
             verdict; 'judge: false' skips it. run prints the resolved job, then the \
             resolved split, then the metrics."
        )),
        positionals: vec![Positional {
            name: "job_file",
            help: "path to the job spec YAML".to_string(),
        }],
        opts: vec![option(
            "--compute-target",
            "COMPUTE_TARGET",
            Kind::Text,
            "export the selected lake rows, submit this run to that canonical \
             Stado compute target, follow it to completion, and run evaluate \
             --best after training"
                .to_string(),
        )],
    };

    let evaluate = Spec {
        name: "evaluate",
        help: "score a trained model on its frozen holdout and have a Brama teacher judge \
               whether the predictions are acceptable"
            .to_string(),
        description: Some(
            "Score the trained model on the frozen holdout in \
             <training root>/models/<name>/eval-split.json — the sessions training \
             never saw — and then send each of them to a Brama-routed teacher with \
             the model's prediction and the ground-truth label, asking whether the \
             prediction is acceptable. The verdict (agreement rate plus one record \
             per session) is written to <training root>/models/<name>/judge.json. A \
             Brama error fails only its own session and is counted; if not one \
             session could be judged, the gateway's own error is reported and the \
             exit status is nonzero — no verdict is invented and there is no local \
             fallback."
                .to_string(),
        ),
        positionals: vec![Positional {
            name: "name",
            help: "job name or aspect — the directory under <training root>/models/".to_string(),
        }],
        opts: vec![
            option(
                "--brama-model",
                "MODEL_ID",
                Kind::Text,
                format!(
                    "Brama-routed judge model (default: the job spec's judge.model, \
                     else {teacher})"
                ),
            ),
            option(
                "--no-judge",
                "",
                Kind::Flag,
                "report the frozen-holdout scores only, without asking the teacher".to_string(),
            ),
            option(
                "--json",
                "",
                Kind::Flag,
                "print machine-readable JSON".to_string(),
            ),
            option(
                "--best",
                "",
                Kind::Flag,
                "after the configured judge, use Brama's best route to audit \
                 whether every ground-truth label and judge opinion is sensible; \
                 exit nonzero when the audit finds an issue"
                    .to_string(),
            ),
        ],
    };

    let infer = Spec {
        name: "infer",
        help: "emit label suggestions for unlabeled sessions".to_string(),
        description: None,
        positionals: Vec::new(),
        opts: vec![
            required(
                "--aspect",
                "ASPECT",
                Kind::Text,
                "aspect name, e.g. reviewed".to_string(),
            ),
            Opt {
                group: 1,
                ..option(
                    "--session",
                    "SESSION",
                    Kind::Text,
                    "predict for one session id, labeled or not".to_string(),
                )
            },
            Opt {
                group: 1,
                ..option(
                    "--limit",
                    "LIMIT",
                    Kind::Int,
                    "cap the number of unlabeled sessions".to_string(),
                )
            },
        ],
    };

    vec![train, run, evaluate, infer]
}
