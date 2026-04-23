import csv
from pathlib import Path

from RAG import get_answer

src = Path("raw_data/вопрос_ответ_par.csv")

out = Path("raw_data/ответы_test.csv")

with src.open("r", encoding="utf-8", newline="") as f_in, out.open("w", encoding="utf-8", newline="") as f_out:
    reader = csv.DictReader(f_in)
    writer = csv.DictWriter(f_out, fieldnames=["N", "question", "pred_answer"])
    writer.writeheader()
    for i, row in enumerate(reader):
        if i >= 10:
            break
        q = (row.get("question") or "").strip()
        writer.writerow({"N": row.get("N", i + 1), "question": q, "pred_answer": get_answer(q)})

print(out)
