#!/usr/bin/env python3
"""Generate deterministic hard-case lifecycle envelopes for Brama review."""

import argparse
import copy
import json
import random
import re
from pathlib import Path


TEMPLATES = {
    "finish_direct": (
        "finishGoal",
        [
            "{title} is fully finished and verified. Close this objective.",
            "Everything requested for {title} now works successfully; this goal is done.",
            "I completed {title} end to end and confirmed the result. Mark it finished.",
            "The work on {title} succeeded in full. Nothing remains for this objective.",
            "{title} jest w pełni skończone i sprawdzone. Zamknij ten cel.",
            "Wszystko dla {title} działa poprawnie; zadanie zostało ukończone.",
        ],
    ),
    # A subagent completion report is not the user confirming completion, which
    # is what the system prompt requires for finishGoal. This family therefore
    # teaches the opposite of what its name used to claim: the objective stays
    # open until the user says otherwise. On 2026-08-17 the served candidate
    # failed the quality gate on exactly these rows — four of its five false
    # completions were subagent notifications.
    "notification_negative": (
        "continueCurrent",
        [
            '<subagent_notification> {{"status":{{"completed":"{title} completed successfully. All requested checks passed and no work remains."}}}}</subagent_notification>',
            '<subagent_notification> {{"status":{{"completed":"Finished {title} end to end; the delivered result is working and verified."}}}}</subagent_notification>',
            '<subagent_notification> {{"status":{{"completed":"Zakończono {title}. Wynik działa poprawnie, został sprawdzony i nic nie pozostało."}}}}</subagent_notification>',
            '<subagent_notification> {{"status":{{"completed":"Updated the requested files for {title} and regenerated the outputs. Compilation succeeded."}}}}</subagent_notification>',
            '<subagent_notification> {{"status":{{"completed":"Completed the requested rounds for {title}; score unchanged. Files changed are listed above."}}}}</subagent_notification>',
        ],
    ),
    "finish_user_handoff": (
        "finishGoal",
        [
            "I am done with {title}; the result works. Please commit and push the finished work.",
            "That completes {title} successfully. Publish the finished changes now.",
            "Skończyłem {title} i wszystko działa. Zacommituj i wypchnij gotową pracę.",
            "We have finished {title} in full. Close it and deliver the completed result.",
        ],
    ),
    "completion_negative": (
        "continueCurrent",
        [
            "The dependency installed successfully, but {title} is still unfinished. Continue the objective.",
            "The command completed and the first check passed. That was only a prerequisite for {title}; keep working.",
            "Good, that sub-step is done. {title} itself is not complete yet, so continue fixing it.",
            "The helper reported success, but the actual objective {title} remains open. Finish the remaining work.",
            "Instalacja się udała, ale {title} nadal nie jest skończone. Kontynuuj pracę.",
            "Ten etap zakończył się sukcesem; cały cel {title} wciąż wymaga dalszej pracy.",
        ],
    ),
    "explicit_open": (
        "continueCurrent",
        [
            "Continue {title} until it is fully done; this objective is explicitly still open.",
            "{title} is not finished and the current result is still broken. Keep working on it.",
            "Do not close {title}. It remains pending; continue monitoring and fixing it.",
            "The previous attempt did not finish {title}. Resume the same objective now.",
            "{title} nadal nie jest skończone. Kontynuuj naprawę aż cel będzie gotowy.",
            "Nie zamykaj {title}; to zadanie pozostaje otwarte i wymaga dalszej pracy.",
        ],
    ),
    "continue_correction": (
        "continueCurrent",
        [
            "That implementation is wrong for {title}. Correct it in the current work instead of starting over.",
            "The result for {title} is missing the required location information. Fix the same objective.",
            "Use the existing mechanism, not the new abstraction, while completing {title}.",
            "This output is unusable for {title}; repair it and preserve the current objective.",
            "To jest błędne dla {title}. Popraw ten sam cel bez tworzenia nowego zadania.",
            "Wynik dla {title} znowu jest bezużyteczny. Napraw go w ramach bieżącej pracy.",
        ],
    ),
    "continue_investigation": (
        "continueCurrent",
        [
            "To advance {title}, inspect the source and explain which implementation is active right now.",
            "Check the logs for {title} and tell me why the current result is missing.",
            "For {title}, compare the two runtime paths and use the one the code actually selects.",
            "Read the current implementation of {title}, identify the failure, and fix it.",
            "Dla {title} sprawdź kod źródłowy i wyjaśnij, dlaczego obecny wynik jest błędny.",
            "Przejrzyj logi {title}, znajdź przyczynę i kontynuuj naprawę tego samego celu.",
        ],
    ),
    "ignore_status": (
        "ignore",
        [
            "What percentage of {title} is done so far? Only report the status.",
            "Where is the current result for {title}? Do not change anything yet.",
            "How did you fix {title}? I only want an explanation of what already happened.",
            "Is {title} currently running or stopped? Just answer the status question.",
            "Jaki jest teraz status {title}? Tylko odpowiedz, niczego nie zmieniaj.",
            "Gdzie jest wynik {title}? Na razie tylko podaj informację.",
        ],
    ),
    "ignore_capability": (
        "ignore",
        [
            "Do you currently have access needed for {title}? Answer yes or no; take no action.",
            "Can the installed tool read the data for {title}? I am asking only about capability.",
            "Which mode is {title} using right now? Do not modify the configuration.",
            "Czy masz dostęp potrzebny do {title}? Tylko odpowiedz, bez wykonywania pracy.",
            "Jakiej ścieżki używa teraz {title}? Nie zmieniaj jeszcze żadnego ustawienia.",
        ],
    ),
    "start_switch": (
        "startGoal",
        [
            "Stop the current work. New task: {new_title}.",
            "Pause that objective; instead, start a separate task to {new_title_lower}.",
            "This is unrelated to the current goal. Begin new work: {new_title}.",
            "Leave the current objective alone. I now want you to {new_title_lower}.",
            "Zatrzymaj bieżącą pracę. Nowe zadanie: {new_title}.",
            "To osobny cel. Teraz zacznij: {new_title_lower}.",
        ],
    ),
}


def read_rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def envelope_for(row):
    return json.loads(next(message["content"] for message in row["messages"] if message["role"] == "user"))


def active_candidate(envelope):
    candidates = [candidate for candidate in envelope.get("candidates", []) if candidate.get("ref") != "NEW_GOAL"]
    same_session = [candidate for candidate in candidates if candidate.get("same_session")]
    pool = same_session or candidates
    if not pool:
        return None
    return max(pool, key=lambda candidate: float(candidate.get("score") or 0.0))


def feature_flags(text, family, turn_index):
    lowered = text.casefold()
    return {
        "turn_index": turn_index,
        "is_first_turn_in_session": False,
        "word_count": len(text.split()),
        "is_short_prompt": len(text.split()) <= 12,
        "is_question": "?" in text,
        "has_new_objective_language": family == "start_switch",
        "has_status_or_meta_language": family.startswith("ignore_"),
        "has_correction_or_rejection_language": family == "continue_correction",
        "has_completion_or_ack_language": family.startswith("finish_")
        or family == "completion_negative"
        or any(token in lowered for token in ("done", "finished", "completed", "skończ", "zakończ")),
    }


def unrelated_title(rows, current_titles, rng):
    for _ in range(100):
        other = envelope_for(rng.choice(rows))
        candidate = active_candidate(other)
        if candidate and candidate.get("title") and candidate["title"].casefold() not in current_titles:
            return candidate["title"]
    return "Audit remote release provenance"


def generate(rows, per_family, seed):
    rng = random.Random(seed)
    eligible = [row for row in rows if active_candidate(envelope_for(row))]
    if not eligible:
        raise SystemExit("source dataset has no lifecycle candidates")
    generated = []
    shuffled = list(eligible)
    rng.shuffle(shuffled)
    cursor = 0
    for family, (intended_action, templates) in TEMPLATES.items():
        for index in range(per_family):
            source = shuffled[cursor % len(shuffled)]
            cursor += 1
            envelope = copy.deepcopy(envelope_for(source))
            candidate = active_candidate(envelope)
            title = candidate["title"]
            current_titles = {
                item.get("title", "").casefold() for item in envelope.get("candidates", [])
            }
            new_title = unrelated_title(eligible, current_titles, rng)
            template = templates[index % len(templates)]
            text = template.format(
                title=title,
                new_title=new_title,
                new_title_lower=new_title[:1].lower() + new_title[1:],
            )
            row_id = f"lifecycle-curriculum-{seed}-{family}-{index + 1:04d}"
            envelope["prompt_id"] = row_id
            envelope["local_day"] = f"curriculum-{seed}"
            envelope["text"] = text
            turn_index = int(envelope.get("turn_index") or 0) + 10_000 + len(generated)
            envelope["turn_index"] = turn_index
            envelope["lifecycle_features"] = feature_flags(text, family, turn_index)
            generated.append(
                {
                    "id": row_id,
                    "split_day": f"curriculum-{seed}",
                    "messages": [
                        {
                            "role": "user",
                            "content": json.dumps(envelope, ensure_ascii=False, separators=(",", ":")),
                        }
                    ],
                    "metadata": {
                        "synthetic": True,
                        "curriculum_family": family,
                        "intended_action": intended_action,
                        "source_id": source["id"],
                        "seed": seed,
                    },
                }
            )
    return generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--per-family", type=int, default=64)
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()
    if args.per_family < 1:
        raise SystemExit("--per-family must be positive")
    generated = generate(read_rows(args.source), args.per_family, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in generated:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(output)
    counts = {family: args.per_family for family in TEMPLATES}
    print(json.dumps({"output": str(output), "rows": len(generated), "families": counts}, sort_keys=True))


if __name__ == "__main__":
    main()
