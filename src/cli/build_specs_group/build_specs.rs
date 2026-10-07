use super::*;

pub(crate) fn build_specs() -> Vec<Spec> {
    let mut specs = training_specs();
    specs.extend(corpus_specs());
    specs.extend(model_specs());
    specs.extend(humanizer_specs());
    specs.extend(release_specs());
    specs.extend(goal_served_specs());
    specs
}

#[allow(unused_variables)]
fn training_specs() -> Vec<Spec> {
    let teacher = brama::DEFAULT_MODEL;
    // Each training key, as the flag `train` reads it; the value is read as a
    // YAML scalar and validated exactly as a job's `training` section is.
    let tfidf = |flag: &'static str, help: &str| {
        option(flag, "VALUE", Kind::Text, format!("tfidf-logreg: {help}; required without --model"))
    };
    let hf = |flag: &'static str, help: &str| {
        option(flag, "VALUE", Kind::Text, format!("HuggingFace: {help}; required with --model"))
    };

    let train_opts = vec![
            required(
                "--aspect",
                "ASPECT",
                Kind::Text,
                "aspect name, e.g. reviewed".to_string(),
            ),
            required(
                "--min-labeled-sessions",
                "N",
                Kind::Int,
                "fewest labeled sessions the training side needs before a model is fitted"
                    .to_string(),
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
        hf("--epochs", "training epochs"),
        hf("--batch-size", "batch size"),
        hf("--learning-rate", "learning rate"),
        hf("--max-length", "tokenizer max tokens per session"),
        hf("--seed", "seed of the head initialisation, in-training slice and shuffle"),
        hf("--weight-decay", "AdamW weight decay, zero or more"),
        hf("--max-grad-norm", "global gradient norm clip, above zero"),
        hf("--in-training-eval-share", "share of the training side sliced off to watch the loss"),
        tfidf("--ngram-max", "longest word n-gram counted"),
        tfidf("--lowercase", "true or false: lowercase text before counting"),
        tfidf("--sublinear-tf", "true or false: term frequency as one plus its logarithm"),
        tfidf("--smooth-idf", "true or false: idf as if one more document held every term"),
        tfidf("--min-df", "terms in fewer documents are dropped"),
        tfidf("--max-df", "terms in more than this share of documents are dropped"),
        tfidf("--c", "inverse strength of the L2 penalty"),
        tfidf("--max-iter", "most L-BFGS iterations"),
        tfidf("--tol", "the solver stops once no gradient component exceeds this"),
        tfidf("--lbfgs-memory", "correction pairs the solver keeps"),
        tfidf("--armijo-c1", "share of the predicted decrease a step must achieve"),
        tfidf("--backtrack", "share a rejected step is shrunk to"),
        tfidf("--max-backtracks", "shrinks tried before a step is abandoned"),
        tfidf("--cv-seed", "seed of each class's cross-validation shuffle"),
            option(
                "--eval-split-fraction",
                "F",
                Kind::Float,
                "share of labeled sessions (between 0 and 1) frozen out of training the \
                 first time this aspect is trained; required unless --no-eval-split. Later \
                 runs reuse the frozen eval-split.json unchanged"
                    .to_string(),
            ),
            option(
                "--eval-split-seed",
                "N",
                Kind::Int,
                "seed that picks the frozen holdout; required unless --no-eval-split".to_string(),
            ),
            option(
                "--no-eval-split",
                "",
                Kind::Flag,
                "train on every labeled session, with no frozen holdout to evaluate on".to_string(),
        ),
    ];
    let train = Spec {
        name: "train",
        help: "train a classifier for one aspect".to_string(),
        description: None,
        positionals: Vec::new(),
        opts: train_opts,
    };

    let run = Spec {
        name: "run",
        help: "execute a declarative training job (YAML spec)".to_string(),
        description: Some(format!(
            "Execute a declarative training job. 'eval_split' is required: a mapping \
             with 'fraction' (between 0 and 1) and 'seed' freezes a holdout of labeled \
             sessions into <training root>/models/<name>/eval-split.json the first time \
             the job runs; every later run reuses that file unchanged, trains on \
             nothing in it, and reports it under 'holdout_evaluation' in metrics.json. \
             'eval_split: false' trains on every labeled session. 'min_labeled_sessions' \
             (required, a positive integer) is the fewest labeled sessions the training \
             side needs before a model is fitted. 'training' is required: for \
             tfidf-logreg ngram_max, lowercase, sublinear_tf, smooth_idf, min_df, max_df, \
             c, max_iter, tol, lbfgs_memory, armijo_c1, backtrack, max_backtracks and \
             cv_seed; for a HuggingFace 'model' epochs, batch_size, learning_rate, \
             max_length, seed, weight_decay, max_grad_norm and in_training_eval_share. \
             'judge' (model: {teacher}) names the Brama-routed \
             teacher that 'evaluate' asks for a verdict; 'judge: false' skips it. run \
             prints the resolved job, then the resolved split, then the metrics."
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
