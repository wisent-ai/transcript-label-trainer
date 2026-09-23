use super::*;

#[allow(unused_variables)]
pub(crate) fn corpus_specs() -> Vec<Spec> {
    let fraction = float_repr(jobs::DEFAULT_EVAL_FRACTION);
    let seed = jobs::DEFAULT_EVAL_SEED;
    let teacher = brama::DEFAULT_MODEL;

    let corpus_adopt = Spec {
        name: "corpus-adopt",
        help: "adopt and select an existing Transcript Lake dataset bundle".to_string(),
        description: Some(
            "Validate the complete schema-v1 dataset bundle before mutation, retain an \
             immutable content-addressed copy under <training root>/corpora, and select it \
             as the input used by train, infer, and evaluate. A repeated import of identical \
             content is unchanged; existing corpora remain retained."
                .to_string(),
        ),
        positionals: vec![Positional {
            name: "bundle",
            help: "dataset-bundle JSON written by a Transcript Label Trainer/Stado export"
                .to_string(),
        }],
        opts: vec![option(
            "--json",
            "",
            Kind::Flag,
            "print machine-readable import counts and retained state".to_string(),
        )],
    };

    let corpus_status = Spec {
        name: "corpus-status",
        help: "show the selected adopted corpus and retained corpus store".to_string(),
        description: None,
        positionals: Vec::new(),
        opts: vec![option(
            "--json",
            "",
            Kind::Flag,
            "print machine-readable JSON".to_string(),
        )],
    };

    let gui = Spec {
        name: "gui",
        help: "serve the graphical corpus importer and HTML documentation".to_string(),
        description: Some(
            "Serve the installed, loopback-only browser workspace for corpus adoption. \
             The GUI uploads one canonical dataset bundle to the same atomic Rust adoption \
             engine as corpus-adopt, then reads back the retained registry. It prints a \
             session-token URL but never opens a browser or starts training, inference, \
             evaluation, teacher/judge work, or a fleet job."
                .to_string(),
        ),
        positionals: Vec::new(),
        opts: vec![
            option(
                "--bind",
                "IP",
                Kind::Text,
                "loopback IP listener (default: 127.0.0.1; also accepts ::1)".to_string(),
            ),
            option(
                "--port",
                "PORT",
                Kind::Int,
                "listener port (default: 0, select an available port)".to_string(),
            ),
        ],
    };

    let info = Spec {
        name: "info",
        help: "list trained aspects, artifacts, and metrics".to_string(),
        description: None,
        positionals: Vec::new(),
        opts: vec![option(
            "--json",
            "",
            Kind::Flag,
            "print machine-readable JSON".to_string(),
        )],
    };

    let onboarding = Spec {
        name: "onboarding",
        help: "walk the published first-use journey to your first suggestions".to_string(),
        description: Some(
            "Walk the first-use journey this repository publishes in \
             onboarding_first_use.json, beginning with an optional existing canonical \
             dataset bundle and ending when a trained classifier emits label suggestions \
             over real transcript text. --corpus calls the same atomic adoption operation \
             as corpus-adopt. Re-running a completed journey reports it complete and \
             changes nothing."
                .to_string(),
        ),
        positionals: Vec::new(),
        opts: vec![
            Opt {
                group: 1,
                ..option(
                    "--corpus",
                    "BUNDLE",
                    Kind::Text,
                    "adopt this existing dataset-bundle JSON at the source-selection step"
                        .to_string(),
                )
            },
            Opt {
                group: 1,
                ..option(
                    "--skip-corpus",
                    "",
                    Kind::Flag,
                    "leave the trainer empty and usable without adopting a corpus".to_string(),
                )
            },
            option(
                "--reset",
                "",
                Kind::Flag,
                "discard the recorded attempt and replay the journey from its entry screen"
                    .to_string(),
            ),
            option(
                "--yes",
                "",
                Kind::Flag,
                "never wait for Enter between screens".to_string(),
            ),
            option(
                "--json",
                "",
                Kind::Flag,
                "print machine-readable JSON".to_string(),
            ),
        ],
    };

    let autolabel = Spec {
        name: "autolabel",
        help: "label every unlabeled session for an aspect via a Brama teacher (zero-touch)"
            .to_string(),
        description: None,
        positionals: Vec::new(),
        opts: vec![
            required(
                "--aspect",
                "ASPECT",
                Kind::Text,
                "aspect name, e.g. tasktype".to_string(),
            ),
            required(
                "--values",
                "VALUES",
                Kind::Text,
                "comma-separated allowed label values, e.g. bugfix,feature,chore,question"
                    .to_string(),
            ),
            option(
                "--brama-model",
                "MODEL_ID",
                Kind::Text,
                format!("Brama-routed teacher model (default: {teacher})"),
            ),
            option(
                "--best",
                "",
                Kind::Flag,
                "have Brama's best route audit every proposed label before it \
                 reaches Transcript Lake; reject nonsensical labels and exit \
                 nonzero when the audit finds an issue"
                    .to_string(),
            ),
            option(
                "--limit",
                "LIMIT",
                Kind::Int,
                "cap the number of sessions labeled".to_string(),
            ),
            option(
                "--runtime",
                "RUNTIME",
                Kind::Text,
                "only sessions of this runtime".to_string(),
            ),
        ],
    };
    vec![corpus_adopt, corpus_status, gui, info, onboarding, autolabel]
}
