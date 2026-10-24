"""ค้นคืน chunk ที่เกี่ยวข้องกับคำถาม (เวิร์กช็อป 2 และ 4) ทำงานในเครื่องทั้งหมด ไม่เรียก API

python -m rag.retrieve "คำถาม" -k 4 [--method tfidf|hybrid] [--include-superseded] [--json]
"""
import argparse
import json
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from rag.chunk import chunk_docs

OUT_KEYS = ["chunk_id", "doc_id", "title", "section", "status", "version", "effective_date", "text", "score"]
METHODS = ["tfidf", "hybrid"]
HYBRID_WEIGHT = 0.5  # สัดส่วนคะแนน TF-IDF ตัวอักษร ส่วนที่เหลือเป็น BM25

# พจนานุกรมคำพ้อง: คำที่ผู้ใช้พิมพ์ → คำทางการในเอกสาร
# สร้างจากคำถามชุด dev ที่ hybrid ยังพลาดอันดับ 1 เท่านั้น ห้ามเพิ่มจากชุด test
SYNONYMS: dict[str, str] = {
    "พาสเวิร์ด": "รหัสผ่าน",  # Q24
    "รถอีวี": "รถยนต์ไฟฟ้า",  # Q35
    "โน้ตบุ๊ก": "คอมพิวเตอร์",  # Q30
    "หาย": "สูญหาย",  # Q30
}


def index_text(c: dict) -> str:
    return f"{c['title']}\n{c['section']}\n{c['text']}"


def tokenize(text: str) -> list[str]:
    from pythainlp.tokenize import word_tokenize

    return [t.lower() for t in word_tokenize(text, engine="newmm", keep_whitespace=False) if t.strip() and any(ch.isalnum() for ch in t)]


def expand_query(query: str) -> str:
    extra = [v for k, v in SYNONYMS.items() if k in query and v not in query]
    return " ".join([query, *extra])


class Retriever:
    def __init__(self, chunks: list[dict] | None = None, method: str = "tfidf"):
        """method="tfidf": TF-IDF ระดับตัวอักษร (W2) · method="hybrid": วิธีที่ปรับปรุงแล้ว (W4)"""
        if method not in METHODS:
            raise ValueError(f"unknown method: {method}")
        self.method = method
        self.chunks = chunk_docs() if chunks is None else chunks
        texts = [index_text(c) for c in self.chunks]
        # ภาษาไทยไม่เว้นวรรคระหว่างคำ จึงนับชิ้นตัวอักษร 2-4 ตัวแทนการตัดคำ
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(texts)
        if method == "hybrid":
            from rank_bm25 import BM25Okapi

            self.bm25 = BM25Okapi([tokenize(t) for t in texts])

    def scores(self, query: str):
        # เวกเตอร์ TF-IDF ถูก normalize แล้ว ผลคูณภายในจึงเท่ากับ cosine similarity
        if self.method == "tfidf":
            return linear_kernel(self.vectorizer.transform([query]), self.matrix).ravel()
        query = expand_query(query)
        tfidf = linear_kernel(self.vectorizer.transform([query]), self.matrix).ravel()
        bm25 = self.bm25.get_scores(tokenize(query))
        top = bm25.max()
        if top > 0:
            bm25 = bm25 / top  # ปรับให้อยู่ช่วง 0-1 เท่ากับ cosine ก่อนรวม
        return HYBRID_WEIGHT * tfidf + (1 - HYBRID_WEIGHT) * bm25

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
    ap.add_argument("--method", default="tfidf", choices=METHODS)
    ap.add_argument("--include-superseded", action="store_true", help="รวมฉบับ superseded (ค่าเริ่มต้นกรองออก)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    res = Retriever(method=a.method).search(a.query, k=a.k, include_superseded=a.include_superseded)
    if a.json:
        sys.stdout.reconfigure(encoding="utf-8")
        print(json.dumps([{k: r[k] for k in OUT_KEYS} for r in res], ensure_ascii=False, indent=1))
        return
    for i, r in enumerate(res, 1):
        print(f"{i}. [{r['chunk_id']}] {r['score']:.3f} · {r['title']} › {r['section']} ({r['status']})")
        print("   " + r["text"].replace("\n", "\n   "))


if __name__ == "__main__":
    main()
