"""Command-line interface: train, infer, info."""

from __future__ import annotations

import argparse
import json
import sys

from . import jobs, model


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcript-label-trainer",
        description=(
            "Train small local classifiers that predict Transcript Lake aspect "
            "labels, and emit label suggestions. Never writes to the lake."
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

    p_run = sub.add_parser("run", help="execute a declarative training job (YAML spec)")
    p_run.add_argument("job_file", help="path to the job spec YAML")

    p_infer = sub.add_parser("infer", help="emit label suggestions for unlabeled sessions")
    p_infer.add_argument("--aspect", required=True, help="aspect name, e.g. reviewed")
    target = p_infer.add_mutually_exclusive_group()
    target.add_argument("--session", help="predict for one session id, labeled or not")
    target.add_argument("--limit", type=int, help="cap the number of unlabeled sessions")

    p_info = sub.add_parser("info", help="list trained aspects, artifacts, and metrics")
    p_info.add_argument("--json", action="store_true", help="print machine-readable JSON")

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
                acc = metrics.get("eval_accuracy")
                quality = f"eval_accuracy={acc}" if acc is not None else "eval_accuracy=n/a"
                quality += f" ({metrics['base_model']}, epochs={hp['epochs']}, lr={hp['lr']}, device={metrics['device']})"
            print(
                f" {marker} {backend}: {metrics['n_sessions']} sessions, "
                f"classes={metrics['classes']}, {quality}\n"
                f"    model:   {metrics['model_path']}\n"
                f"    trained: {metrics['trained_at']}"
            )


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.command == "train":
        try:
            metrics = model.train(
                args.aspect,
                model_id=args.model_id,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                max_length=args.max_length,
            )
        except model.NotEnoughData as exc:
            sys.stderr.write(f"train: {exc}\n")
            sys.exit(2)
        except (ValueError, RuntimeError, model.HfExtraMissing) as exc:
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
            metrics = model.run_job(job, resolved)
        except model.NotEnoughData as exc:
            sys.stderr.write(f"run: {exc}\n")
            sys.exit(2)
        except (ValueError, RuntimeError, model.HfExtraMissing) as exc:
            sys.stderr.write(f"run: {exc}\n")
            sys.exit(1)
        print(json.dumps(metrics, indent=2))
        return

    if args.command == "infer":
        try:
            suggestions = model.infer(args.aspect, session=args.session, limit=args.limit)
        except (ValueError, FileNotFoundError, RuntimeError, model.HfExtraMissing) as exc:
            sys.stderr.write(f"infer: {exc}\n")
            sys.exit(1)
        print(json.dumps(suggestions, indent=2))
        return

    if args.command == "info":
        entries = model.info()
        if args.json:
            print(json.dumps(entries, indent=2))
            return
        _print_info(entries)
        return


if __name__ == "__main__":
    main()
