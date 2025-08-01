# prepare_dataset.py
import json
from pathlib import Path

source = Path(".")
output = Path("dostoevsky_dataset.jsonl")

with output.open("w", encoding="utf-8") as out:
    for file in source.glob("*.txt"):
        text = file.read_text(encoding="utf-8").strip()
        out.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

