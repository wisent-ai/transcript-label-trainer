#!/usr/bin/env python3
"""Remove synthetic tool probes and add multilingual task-title examples."""

import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
synthetic_prefixes = (
    "use the read tool",
    "use bash to run exactly",
    "use the monitor tool",
    "use the notebookedit tool",
    "use the edit tool",
)
examples = [
    ("czy Brama ma aplikację na komputer?", "Sprawdź aplikację desktopową Brama"),
    ("sprawdź, czy Skarbiec ma aplikację desktopową", "Sprawdź aplikację desktopową Skarbiec"),
    ("czy Brama i Skarbiec mają własne aplikacje na macOS?", "Sprawdź aplikacje Brama i Skarbiec"),
    ("znajdź transkrypt, w którym omawialiśmy upload z RTX boxa", "Znajdź transkrypt o uploadzie RTX boxa"),
    ("nie mogę znaleźć rozmowy o wysyłaniu danych z RTX boxa", "Znajdź rozmowę o uploadzie RTX boxa"),
    ("odszukaj rozmowę o wdrożeniu na naszym RTX boxie", "Odszukaj rozmowę o wdrożeniu RTX"),
    ("znajdź wczorajszą rozmowę o dynamicznych interfejsach i kontynuuj pracę", "Znajdź i kontynuuj rozmowę o interfejsach"),
    ("pamiętasz dyskusję o dynamicznym UI na iOS? znajdź ją i kontynuuj", "Znajdź i kontynuuj dyskusję o iOS"),
    ("odszukaj wcześniejszą sesję o interfejsach iOS i wróć do pracy", "Odszukaj i kontynuuj sesję o iOS"),
    ("straciliśmy sesje OMP, znajdź je i uruchom ponownie w agentach", "Znajdź i wznów sesje OMP"),
    ("sprawdź utracone sesje OMP i wznów je w dodatkowych agentach", "Sprawdź i wznów sesje OMP"),
    ("ustal co zamknęło agentów i napraw problem", "Zdiagnozuj i napraw zamknięte agenty"),
    ("dlaczego agenty przestały działać? znajdź przyczynę i je napraw", "Zdiagnozuj i napraw agenty"),
    ("czy Lem ma natywną aplikację na Maca?", "Sprawdź aplikację desktopową Lem"),
    ("odszukaj transkrypt o publikowaniu artefaktów z serwera GPU", "Odszukaj transkrypt o publikowaniu artefaktów"),
    ("znajdź starą rozmowę o adaptacyjnym interfejsie i kontynuuj implementację", "Znajdź i kontynuuj rozmowę o interfejsie"),
]

rows = []
for raw in source.read_text(encoding="utf-8").splitlines():
    if not raw.strip():
        continue
    row = json.loads(raw)
    if row.get("message", "").strip().lower().startswith(synthetic_prefixes):
        continue
    rows.append(row)
for index, (message, goal) in enumerate(examples):
    rows.append({
        "session_id": f"curated-multilingual-{index:03d}",
        "runtime": "curated",
        "message": message,
        "goal": goal,
        "goal_source": "curated:multilingual-task-contract-v1",
        "gold": False,
        "reviewed_by": "curated:multilingual-task-contract-v1",
    })
destination.write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    encoding="utf-8",
)
print(json.dumps({
    "source_rows": len(rows) - len(examples),
    "curated_rows": len(examples),
    "total_rows": len(rows),
}, sort_keys=True))
