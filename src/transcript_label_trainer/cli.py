"""Command-line interface: train, run, infer, evaluate, info, autolabel."""

from __future__ import annotations

import argparse
import json
import sys

from . import autolabel, brama, evaluate, jobs, model, placement


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcript-label-trainer",
        description=(
            "Train small local classifiers that predict Transcript Lake aspect "
            "labels, and emit label suggestions. Never writes to the lake."
        ),
    )
    parser.add_argument(
        "--training-root",
        metavar="PATH",
        help=(
            "override where model artifacts live; beats $TLT_HOME and the "
            "Stado registry declaration"
        ),
    )
    parser.add_argument(
        "--storage-root",
        metavar="PATH",
        help=(
            "override the lake data root; beats $LAKE_DATA and the Stado "
            "registry declaration"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="train a classifier for one aspect")
    p_train.add_argument("--aspect", required=True, help="aspect name, e.g. reviewed")
    p_train.add_argument(
        "--model",
        dest="model_id",
        metavar="HF_MODEL_ID",
        help=(
            "fine-tune this HuggingFace model instead of the default TF-IDF + "
            "logistic regression (requires the 'hf' extra); multilingual models "
            "such as distilbert-base-multilingual-cased fit the mixed "
            "Polish/English transcripts"
        ),
    )
    p_train.add_argument("--epochs", type=float, default=3, help="HF training epochs (default: 3)")
    p_train.add_argument("--batch-size", type=int, default=8, help="HF batch size (default: 8)")
    p_train.add_argument("--lr", type=float, default=2e-5, help="HF learning rate (default: 2e-5)")
    p_train.add_argument(
        "--max-length", type=int, default=512, help="HF tokenizer max tokens per session (default: 512)"
    )
    p_train.add_argument(
        "--eval-split-fraction",
        type=float,
        default=jobs.DEFAULT_EVAL_FRACTION,
        metavar="F",
        help=(
            "share of labeled sessions frozen out of training the first time "
            f"this aspect is trained (default: {jobs.DEFAULT_EVAL_FRACTION}); "
            "later runs reuse the frozen eval-split.json unchanged"
        ),
    )
    p_train.add_argument(
        "--eval-split-seed",
        type=int,
        default=jobs.DEFAULT_EVAL_SEED,
        metavar="N",
        help=f"seed that picks the frozen holdout (default: {jobs.DEFAULT_EVAL_SEED})",
    )
    p_train.add_argument(
        "--no-eval-split",
        action="store_true",
        help="train on every labeled session, with no frozen holdout to evaluate on",
    )

    p_run = sub.add_parser(
        "run",
        help="execute a declarative training job (YAML spec)",
        description=(
            "Execute a declarative training job. Two spec sections are on "
            "unless the spec turns them off. 'eval_split' (fraction: "
            f"{jobs.DEFAULT_EVAL_FRACTION}, seed: {jobs.DEFAULT_EVAL_SEED}) "
            "freezes a holdout of labeled sessions into "
            "<training root>/models/<name>/eval-split.json the first time the "
            "job runs; every later run reuses that file unchanged, trains on "
            "nothing in it, and reports it under 'holdout_evaluation' in "
            "metrics.json. 'eval_split: false' trains on every labeled "
            "session. 'judge' (model: "
            f"{brama.DEFAULT_MODEL}) names the Brama-routed teacher that "
            "'evaluate' asks for a verdict; 'judge: false' skips it. run "
            "prints the resolved job, then the resolved split, then the "
            "metrics."
        ),
    )
    p_run.add_argument("job_file", help="path to the job spec YAML")

    p_eval = sub.add_parser(
        "evaluate",
        help=(
            "score a trained model on its frozen holdout and have a Brama "
            "teacher judge whether the predictions are acceptable"
        ),
        description=(
            "Score the trained model on the frozen holdout in "
            "<training root>/models/<name>/eval-split.json — the sessions "
            "training never saw — and then send each of them to a "
            "Brama-routed teacher with the model's prediction and the "
            "ground-truth label, asking whether the prediction is acceptable. "
            "The verdict (agreement rate plus one record per session) is "
            "written to <training root>/models/<name>/judge.json. A Brama "
            "error fails only its own session and is counted; if not one "
            "session could be judged, the gateway's own error is reported and "
            "the exit status is nonzero — no verdict is invented and there is "
            "no local fallback."
        ),
    )
    p_eval.add_argument(
        "name",
        help="job name or aspect — the directory under <training root>/models/",
    )
    p_eval.add_argument(
        "--brama-model",
        metavar="MODEL_ID",
        help=(
            "Brama-routed judge model (default: the job spec's judge.model, "
            f"else {brama.DEFAULT_MODEL})"
        ),
    )
    p_eval.add_argument(
        "--no-judge",
        action="store_true",
        help="report the frozen-holdout scores only, without asking the teacher",
    )
    p_eval.add_argument("--json", action="store_true", help="print machine-readable JSON")

    p_infer = sub.add_parser("infer", help="emit label suggestions for unlabeled sessions")
    p_infer.add_argument("--aspect", required=True, help="aspect name, e.g. reviewed")
    target = p_infer.add_mutually_exclusive_group()
    target.add_argument("--session", help="predict for one session id, labeled or not")
    target.add_argument("--limit", type=int, help="cap the number of unlabeled sessions")

    p_info = sub.add_parser("info", help="list trained aspects, artifacts, and metrics")
    p_info.add_argument("--json", action="store_true", help="print machine-readable JSON")

    p_auto = sub.add_parser(
        "autolabel",
        help="label every unlabeled session for an aspect via a Brama teacher (zero-touch)",
    )
    p_auto.add_argument("--aspect", required=True, help="aspect name, e.g. tasktype")
    p_auto.add_argument(
        "--values",
        required=True,
        help="comma-separated allowed label values, e.g. bugfix,feature,chore,question",
    )
    p_auto.add_argument(
        "--brama-model",
        metavar="MODEL_ID",
        help=f"Brama-routed teacher model (default: {brama.DEFAULT_MODEL})",
    )
    p_auto.add_argument("--limit", type=int, help="cap the number of sessions labeled")
    p_auto.add_argument("--runtime", help="only sessions of this runtime")

    return parser


def _print_info(entries: list[dict]) -> None:
    if not entries:
        print(f"no trained aspects under {model.models_dir()}")
        return
    for entry in entries:
        artifacts = entry["artifacts"]
        if not artifacts:
            print(f"{entry['aspect']}: no trained artifacts in {entry['dir']}")
            continue
        print(f"{entry['aspect']} (active backend: {entry['active']}):")
        for artifact in artifacts:
            metrics = artifact["metrics"]
            backend = artifact["backend"]
            marker = "*" if backend == entry["active"] else " "
            job = metrics.get("job")
            if job:
                print(f"    job:     {job['name']} — {job['task']} (evaluator: {job['evaluator']})")
            if backend == "sklearn":
                cv = metrics["cv_accuracy"]
                quality = (
                    f"cv_accuracy={cv} ({metrics['cv_folds']}-fold)" if cv is not None else "cv_accuracy=n/a"
                )
            else:
                hp = metrics["hyperparameters"]
                acc = metrics.get("in_training_eval", {}).get("accuracy")
                quality = f"in_training_accuracy={acc}" if acc is not None else "in_training_accuracy=n/a"
                quality += f" ({metrics['base_model']}, epochs={hp['epochs']}, lr={hp['lr']}, device={metrics['device']})"
            print(
                f" {marker} {backend}: {metrics['n_sessions']} sessions trained on, "
                f"classes={metrics['classes']}, {quality}\n"
                f"    model:   {metrics['model_path']}\n"
                f"    trained: {metrics['trained_at']}"
            )
            holdout = metrics.get("holdout_evaluation")
            split = metrics.get("eval_split") or {}
            if holdout:
                print(
                    f"    frozen holdout: accuracy={holdout['accuracy']} on "
                    f"{holdout['n_sessions']} session(s), "
                    f"seed={split.get('seed')}, fraction={split.get('fraction')}"
                )
            elif split and not split.get("enabled", True):
                print("    frozen holdout: disabled by eval_split: false")


def _print_placement() -> None:
    """Where Stado puts this run — and, when it could not, why it is local."""
    resolved = placement.resolve_placement()
    print("placement:")
    print(f"    source:        {resolved.source}")
    print(f"    training host: {resolved.training_host or 'undeclared'}")
    print(f"    training root: {resolved.training_root}")
    print(f"    storage root:  {resolved.storage_root}")
    if resolved.source == "local-fallback":
        print(f"    fallback:      {resolved.detail}")
    print()


def _print_verdict(verdict: dict) -> None:
    """The evaluate report: frozen-holdout scores, then the teacher's verdict."""
    split = verdict["eval_split"]
    holdout = verdict["holdout_evaluation"]
    print(f"{verdict['name']} (aspect: {verdict['aspect']}, backend: {verdict['backend']}):")
    print(
        f"    frozen split:  {split['frozen_sessions']} session(s), "
        f"fraction={split['fraction']}, seed={split['seed']}, "
        f"created {split['created_at']}\n"
        f"    split file:    {split['path']}"
    )
    if split["missing_ground_truth"]:
        print(f"    unlabeled now: {split['missing_ground_truth']} (excluded)")
    if split["skipped_no_text"]:
        print(f"    without text:  {split['skipped_no_text']} (excluded)")
    print(
        f"    holdout:       accuracy={holdout['accuracy']} on "
        f"{holdout['n_sessions']} session(s)"
    )
    for value in sorted(holdout["counts"]):
        print(
            f"        {value}: {holdout['correct'].get(value, 0)}/"
            f"{holdout['counts'][value]} correct"
        )
    for pair in holdout["confusion"]:
        print(f"        confused {pair['gold']} -> {pair['predicted']} ({pair['n']}x)")
    judge = verdict["judge"]
    if not judge["enabled"]:
        print("    judge:         skipped (--no-judge)")
        return
    print(
        f"    judge:         {judge['model']} calls "
        f"{judge['acceptable']}/{judge['judged']} prediction(s) acceptable "
        f"(agreement_rate={judge['agreement_rate']}, failed={judge['failed']})"
    )
    for record in verdict["sessions"]:
        mark = "ok " if record["verdict"] == evaluate.JUDGE_VALUES[0] else "bad"
        print(
            f"        {mark} {record['session_id']}: gold={record['gold']} "
            f"predicted={record['prediction']} ({record['confidence']})"
        )
    for failure in verdict["failures"]:
        print(f"        err {failure['session_id']}: {failure['error']}")
    print(f"    verdict file:  {verdict['judge_path']}")


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    placement.set_override(args.training_root, args.storage_root)

    if args.command == "train":
        eval_split = (
            {"enabled": False, "fraction": None, "seed": None}
            if args.no_eval_split
            else {
                "enabled": True,
                "fraction": args.eval_split_fraction,
                "seed": args.eval_split_seed,
            }
        )
        try:
            metrics = model.train(
                args.aspect,
                model_id=args.model_id,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                max_length=args.max_length,
                eval_split=eval_split,
            )
        except model.NotEnoughData as exc:
            sys.stderr.write(f"train: {exc}\n")
            sys.exit(2)
        except (ValueError, RuntimeError, model.HfExtraMissing, evaluate.SplitError) as exc:
            sys.stderr.write(f"train: {exc}\n")
            sys.exit(1)
        print(json.dumps(metrics, indent=2))
        return

    if args.command == "run":
        try:
            job = jobs.load(args.job_file)
        except jobs.JobError as exc:
            sys.stderr.write(f"run: {exc}\n")
            sys.exit(1)
        resolved = model.resolve_job(job)
        print(json.dumps(model.job_summary(job, resolved), indent=2))
        try:
            plan = model.prepare_job(job, resolved)
        except model.NotEnoughData as exc:
            sys.stderr.write(f"run: {exc}\n")
            sys.exit(2)
        except (ValueError, RuntimeError, evaluate.SplitError) as exc:
            sys.stderr.write(f"run: {exc}\n")
            sys.exit(1)
        print(json.dumps({"eval_split": model.split_summary(plan)}, indent=2))
        try:
            metrics = model.run_job(job, plan)
        except model.NotEnoughData as exc:
            sys.stderr.write(f"run: {exc}\n")
            sys.exit(2)
        except (ValueError, RuntimeError, model.HfExtraMissing) as exc:
            sys.stderr.write(f"run: {exc}\n")
            sys.exit(1)
        print(json.dumps(metrics, indent=2))
        return

    if args.command == "evaluate":
        try:
            verdict = evaluate.evaluate(
                args.name,
                judge=False if args.no_judge else None,
                judge_model=args.brama_model,
            )
        except (
            ValueError,
            FileNotFoundError,
            RuntimeError,
            evaluate.SplitError,
            brama.BramaError,
            model.HfExtraMissing,
        ) as exc:
            sys.stderr.write(f"evaluate: {exc}\n")
            sys.exit(1)
        if args.json:
            print(json.dumps(verdict, indent=2))
            return
        _print_verdict(verdict)
        return

    if args.command == "infer":
        try:
            suggestions = model.infer(args.aspect, session=args.session, limit=args.limit)
        except (ValueError, FileNotFoundError, RuntimeError, model.HfExtraMissing) as exc:
            sys.stderr.write(f"infer: {exc}\n")
            sys.exit(1)
        print(json.dumps(suggestions, indent=2))
        return

    if args.command == "autolabel":
        values = [v.strip() for v in args.values.split(",") if v.strip()]
        if not values:
            sys.stderr.write("autolabel: --values must name at least one allowed value\n")
            sys.exit(1)
        try:
            summary = autolabel.autolabel(
                args.aspect,
                values,
                brama_model=args.brama_model,
                limit=args.limit,
                runtime=args.runtime,
            )
        except (ValueError, RuntimeError, brama.BramaError) as exc:
            sys.stderr.write(f"autolabel: {exc}\n")
            sys.exit(1)
        print(json.dumps(summary, indent=2))
        return

    if args.command == "info":
        entries = model.info()
        if args.json:
            print(json.dumps({"placement": placement.as_dict(), "aspects": entries}, indent=2))
            return
        _print_placement()
        _print_info(entries)
        return


if __name__ == "__main__":
    main()
