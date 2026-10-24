"""แบ่งเอกสารเป็น chunk พร้อม metadata (เวิร์กช็อป 1)

แต่ละ chunk เป็น dict ที่มีอย่างน้อย:
chunk_id ("<doc_id>#<ลำดับ>"), doc_id, title, section, status, effective_date, text
"""
from rag.ingest import load_docs

MAX_CHARS = 600
META_KEYS = ["doc_id", "title", "department", "version", "effective_date", "status", "path"]


def split_sections(body: str) -> list[tuple[str, str]]:
    """คืน [(ชื่อหัวข้อ ##, เนื้อหา)] ข้ามข้อความก่อนหัวข้อ ## แรก (เช่นบรรทัด # ชื่อเรื่อง)"""
    sections, name, lines = [], None, []
    for line in body.splitlines():
        if line.startswith("## "):
            if name is not None:
                sections.append((name, "\n".join(lines).strip()))
            name, lines = line[3:].strip(), []
        elif name is not None:
            lines.append(line)
    if name is not None:
        sections.append((name, "\n".join(lines).strip()))
    return sections


def split_long(text: str, max_chars: int) -> list[str]:
    """ตัดข้อความที่ยาวเกิน max_chars ตามบรรทัด (บรรทัดเดียวที่ยาวเกินจึงตัดกลางบรรทัด)"""
    parts, cur = [], ""
    for line in text.splitlines():
        while len(line) > max_chars:
            if cur:
                parts.append(cur)
                cur = ""
            parts.append(line[:max_chars])
            line = line[max_chars:]
        if cur and len(cur) + 1 + len(line) > max_chars:
            parts.append(cur)
            cur = line
        else:
            cur = f"{cur}\n{line}" if cur else line
    if cur.strip():
        parts.append(cur)
    return [p for p in parts if p.strip()]


def chunk_docs(docs: list[dict] | None = None, max_chars: int = MAX_CHARS) -> list[dict]:
    docs = load_docs() if docs is None else docs
    chunks = []
    for d in docs:
        meta = {k: d.get(k, "") for k in META_KEYS}
        n = 0
        for section, text in split_sections(d["body"]):
            for part in split_long(text, max_chars):
                n += 1
                chunks.append({**meta, "chunk_id": f"{d['doc_id']}#{n}", "section": section, "text": part})
    return chunks


if __name__ == "__main__":
    cs = chunk_docs()
    print(len(cs), "chunks · ยาวสุด", max(len(c["text"]) for c in cs), "ตัวอักษร")
