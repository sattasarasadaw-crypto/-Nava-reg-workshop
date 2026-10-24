"""ค้นคืน chunk ที่เกี่ยวข้องกับคำถาม (เวิร์กช็อป 2 และ 4) ทำงานในเครื่องทั้งหมด ไม่เรียก API

python -m rag.retrieve "คำถาม" -k 4 [--method tfidf|hybrid] [--active-only] [--json]
"""
import argparse
import json
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from rag.chunk import chunk_docs

OUT_KEYS = ["chunk_id", "doc_id", "title", "section", "status", "version", "effective_date", "text", "score"]


def index_text(c: dict) -> str:
    return f"{c['title']}\n{c['section']}\n{c['text']}"


class Retriever:
    def __init__(self, chunks: list[dict] | None = None, method: str = "tfidf"):
        """method="tfidf": TF-IDF ระดับตัวอักษร (W2) · method="hybrid": วิธีที่ปรับปรุงแล้ว (W4)"""
        if method != "tfidf":
            raise ValueError(f"unknown method: {method}")
        self.method = method
        self.chunks = chunk_docs() if chunks is None else chunks
        # ภาษาไทยไม่เว้นวรรคระหว่างคำ จึงนับชิ้นตัวอักษร 2-4 ตัวแทนการตัดคำ
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([index_text(c) for c in self.chunks])

    def scores(self, query: str):
        # เวกเตอร์ TF-IDF ถูก normalize แล้ว ผลคูณภายในจึงเท่ากับ cosine similarity
        return linear_kernel(self.vectorizer.transform([query]), self.matrix).ravel()

    def search(self, query: str, k: int = 4, include_superseded: bool = True) -> list[dict]:
        """คืน chunk k อันที่คะแนนสูงสุด เรียงจากมากไปน้อย แต่ละอันเพิ่มช่อง "score" """
        s = self.scores(query)
        order = sorted(range(len(self.chunks)), key=lambda i: -s[i])
        res = []
        for i in order:
            c = self.chunks[i]
            if not include_superseded and c["status"] != "active":
                continue
            res.append({**c, "score": float(s[i])})
            if len(res) == k:
                break
        return res


def main(argv=None):
    ap = argparse.ArgumentParser(description="ค้นคืน chunk จากเอกสารบริษัท")
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=4)
    ap.add_argument("--method", default="tfidf")
    ap.add_argument("--active-only", action="store_true", help="กรองฉบับ superseded ออก")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    res = Retriever(method=a.method).search(a.query, k=a.k, include_superseded=not a.active_only)
    if a.json:
        sys.stdout.reconfigure(encoding="utf-8")
        print(json.dumps([{k: r[k] for k in OUT_KEYS} for r in res], ensure_ascii=False, indent=1))
        return
    for i, r in enumerate(res, 1):
        print(f"{i}. [{r['chunk_id']}] {r['score']:.3f} · {r['title']} › {r['section']} ({r['status']})")
        print("   " + r["text"].replace("\n", "\n   "))


if __name__ == "__main__":
    main()
