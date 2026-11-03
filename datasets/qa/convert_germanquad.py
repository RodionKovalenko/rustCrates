"""GermanQuAD parquet (Hugging Face refs/convert/parquet) -> SQuAD JSON.
Context: the title line is removed (it becomes the article title), "=== Section ===" becomes "Section".
Contexts longer than MAX_CHARS are skipped (XQuAD's longest is 3884)."""
import json, re, sys
import pyarrow.parquet as pq
MAX_CHARS = 4000
src, dst = sys.argv[1], sys.argv[2]
articles, skipped, questions = {}, 0, 0
for r in pq.read_table(src).to_pylist():
    title, _, body = r["context"].partition("\n\n")
    body = re.sub(r"^=+\s*(.*?)\s*=+\s*$", r"\1", body.strip(), flags=re.M)
    answer = next((a.strip() for a in r["answers"]["text"] if a.strip()), None)
    if not answer:
        skipped += 1; continue
    if len(body) > MAX_CHARS:
        skipped += 1; continue
    paragraphs = articles.setdefault(title.replace("_", " ").strip(), {})
    p = paragraphs.setdefault(body, {"context": body, "qas": []})
    p["qas"].append({"id": str(r["id"]), "question": r["question"].strip(), "answers": [{"text": answer, "answer_start": max(body.find(answer), 0)}]})
    questions += 1
data = [{"title": t, "paragraphs": list(ps.values())} for t, ps in articles.items()]
json.dump({"version": "GermanQuAD", "data": data}, open(dst, "w"), ensure_ascii=False)
print(dst, questions, "questions", sum(len(a["paragraphs"]) for a in data), "paragraphs; skipped", skipped)
