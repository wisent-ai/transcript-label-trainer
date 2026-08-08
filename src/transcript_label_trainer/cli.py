"""Command-line interface: train, infer, info."""

from __future__ import annotations

import argparse
import json
import sys

from . import model


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

    p_infer = sub.add_parser("infer", help="emit label suggestions for unlabeled sessions")
    p_infer.add_argument("--aspect", required=True, help="aspect name, e.g. reviewed")
    target = p_infer.add_mutually_exclusive_group()
    target.add_argument("--session", help="predict for one session id, labeled or not")
    target.add_argument("--limit", type=int, help="cap the number of unlabeled sessions")

    p_info = sub.add_parser("info", help="list trained aspects, artifacts, and metrics")
    p_info.add_argument("--json", action="store_true", help="print machine-readable JSON")

    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.command == "train":
        try:
            metrics = model.train(args.aspect)
        except model.NotEnoughData as exc:
            sys.stderr.write(f"train: {exc}\n")
            sys.exit(2)
        except (ValueError, RuntimeError) as exc:
            sys.stderr.write(f"train: {exc}\n")
            sys.exit(1)
        print(json.dumps(metrics, indent=2))
        return

    if args.command == "infer":
        try:
            suggestions = model.infer(args.aspect, session=args.session, limit=args.limit)
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            sys.stderr.write(f"infer: {exc}\n")
            sys.exit(1)
        print(json.dumps(suggestions, indent=2))
        return

    if args.command == "info":
        entries = model.info()
        if args.json:
            print(json.dumps(entries, indent=2))
            return
        if not entries:
            print(f"no trained aspects under {model.models_dir()}")
            return
        for entry in entries:
            metrics = entry["metrics"]
            if metrics is None:
                print(f"{entry['aspect']}: no metrics.json in {entry['dir']}")
                continue
            cv = metrics["cv_accuracy"]
            cv_text = f"cv_accuracy={cv} ({metrics['cv_folds']}-fold)" if cv is not None else "cv_accuracy=n/a"
            print(
                f"{entry['aspect']}: {metrics['n_sessions']} sessions, "
                f"classes={metrics['classes']}, {cv_text}\n"
                f"  model:   {metrics['model_path']}\n"
                f"  metrics: {entry['dir']}/metrics.json\n"
                f"  trained: {metrics['trained_at']}"
            )
        return


if __name__ == "__main__":
    main()
