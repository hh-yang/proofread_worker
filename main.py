import io
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import requests
from PIL import Image
import pytesseract
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field
from openai import OpenAI

# ----------------------------
# ENV
# ----------------------------
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

WORKER_API_KEY = os.getenv("WORKER_API_KEY", "")  # required header for all POSTs

# Tunables
TEXT_MIN_CHARS_PER_PAGE = int(os.getenv("TEXT_MIN_CHARS_PER_PAGE", "30"))
OCR_DPI = int(os.getenv("OCR_DPI", "200"))
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "1800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
MAX_EVIDENCE_BLOCKS = int(os.getenv("MAX_EVIDENCE_BLOCKS", "350"))

oai = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI(title="ProofRead Worker (Free OCR, No Supabase Server Key)")


# ----------------------------
# DTOs
# ----------------------------
class ProcessUrlRequest(BaseModel):
    doc_id: str
    pdf_url: str
    embed: bool = True              # if True, returns embeddings for chunks
    include_figures: bool = True    # if True, returns figure bbox blocks

class EvidenceBlock(BaseModel):
    client_id: str
    page_number: int
    block_type: str               # line | figure | table_cell (if you add later)
    text: str = ""
    bbox_norm: Dict[str, float]   # {x0,y0,x1,y1}

class AskRequest(BaseModel):
    doc_id: str
    mode: str = "chat"
    question: Optional[str] = None
    chat_history: List[Dict[str, str]] = Field(default_factory=list)
    evidence_blocks: List[EvidenceBlock] = Field(default_factory=list)
    ignored_sections: List[str] = Field(default_factory=list)


# ----------------------------
# Security
# ----------------------------
def require_worker_key(x_worker_key: Optional[str]):
    if WORKER_API_KEY and x_worker_key != WORKER_API_KEY:
        raise HTTPException(401, "Invalid worker key")


# ----------------------------
# Utils
# ----------------------------
def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def bbox_norm_from_abs(x0: float, y0: float, x1: float, y1: float, w: float, h: float) -> Dict[str, float]:
    return {
        "x0": clamp01(x0 / w),
        "y0": clamp01(y0 / h),
        "x1": clamp01(x1 / w),
        "y1": clamp01(y1 / h),
    }

def union_bbox(a: Tuple[float,float,float,float], b: Tuple[float,float,float,float]) -> Tuple[float,float,float,float]:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))

def sentence_split(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p.strip() for p in parts if p.strip()]

def compute_strength(claim: str, snippet: str) -> str:
    claim_l = claim.lower()
    snip_l = snippet.lower()
    overlap = sum(1 for w in set(re.findall(r"[a-z0-9]+", claim_l)) if w in snip_l)
    if overlap >= 6:
        return "High"
    if overlap >= 3:
        return "Medium"
    return "Low"

def embed_texts(texts: List[str]) -> List[List[float]]:
    resp = oai.embeddings.create(model=OPENAI_EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def extract_text_lines_with_bboxes(page: fitz.Page) -> List[Dict[str, Any]]:
    """
    Extract selectable text lines with bounding boxes from the PDF text layer.
    Returns [{text, bbox_abs:(x0,y0,x1,y1)}]
    """
    d = page.get_text("dict")
    out = []
    for block in d.get("blocks", []):
        if block.get("type") != 0:  # 0 = text
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            text = "".join(s.get("text", "") for s in spans).strip()
            if not text:
                continue
            bbox = line.get("bbox") or spans[0].get("bbox")
            if bbox and len(bbox) == 4:
                out.append({"text": text, "bbox_abs": (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))})
    return out

def ocr_page_lines(page: fitz.Page, dpi: int) -> List[Dict[str, Any]]:
    """
    OCR page as an image using Tesseract.
    Returns [{text, bbox_abs:(x0,y0,x1,y1), confidence}]
    """
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png")))

    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

    # group by line
    groups: Dict[Tuple[int,int,int], Dict[str, Any]] = {}
    n = len(data["text"])
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue

        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        left, top = float(data["left"][i]), float(data["top"][i])
        w, h = float(data["width"][i]), float(data["height"][i])
        x0, y0, x1, y1 = left, top, left + w, top + h

        try:
            conf_val = float(data["conf"][i])
        except Exception:
            conf_val = None

        if key not in groups:
            groups[key] = {"parts": [], "bbox_px": (x0,y0,x1,y1), "conf": []}

        groups[key]["parts"].append(txt)
        groups[key]["bbox_px"] = union_bbox(groups[key]["bbox_px"], (x0,y0,x1,y1))
        if conf_val is not None and conf_val >= 0:
            groups[key]["conf"].append(conf_val)

    blocks = []
    for g in groups.values():
        text = " ".join(g["parts"]).strip()
        (px0, py0, px1, py1) = g["bbox_px"]
        # px -> page coords
        x0, y0, x1, y1 = px0 / zoom, py0 / zoom, px1 / zoom, py1 / zoom
        conf = (sum(g["conf"]) / len(g["conf"])) if g["conf"] else None
        blocks.append({"text": text, "bbox_abs": (x0,y0,x1,y1), "confidence": conf})
    return blocks

def extract_figures(page: fitz.Page) -> List[Dict[str, Any]]:
    """
    Extract figure bounding boxes (embedded images). Returns bbox_abs list.
    (We do NOT upload/store images here; Lovable can if needed.)
    """
    figs = []
    imgs = page.get_images(full=True)
    for img in imgs:
        xref = img[0]
        rects = page.get_image_rects(xref)
        if rects:
            r = rects[0]
            figs.append({"bbox_abs": (float(r.x0), float(r.y0), float(r.x1), float(r.y1))})
    return figs

def build_chunks_from_page_text(page_texts: Dict[int, str]) -> List[Dict[str, Any]]:
    chunks = []
    for p in sorted(page_texts.keys()):
        text = (page_texts[p] or "").strip()
        if not text:
            continue
        start = 0
        while start < len(text):
            end = min(len(text), start + CHUNK_MAX_CHARS)
            chunks.append({
                "client_id": f"p{p}-c{len(chunks)}",
                "page_start": p,
                "page_end": p,
                "content": text[start:end],
            })
            if end == len(text):
                break
            start = max(0, end - CHUNK_OVERLAP)
    return chunks

def call_openai_json(system: str, user: str) -> Dict[str, Any]:
    r = oai.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    raw = r.choices[0].message.content or "{}"
    try:
        return json.loads(raw)
    except Exception:
        return {"answer_text": "Failed to parse model output.", "citations": [], "evidence_cards": [], "coverage": {"grounded_pct": 0, "unsupported_sentences": []}, "why_sources": {}}


# ----------------------------
# Routes
# ----------------------------
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/process_url")
def process_url(req: ProcessUrlRequest, x_worker_key: Optional[str] = Header(default=None)):
    require_worker_key(x_worker_key)

    # Download PDF bytes
    resp = requests.get(req.pdf_url, timeout=60)
    if resp.status_code != 200:
        raise HTTPException(400, f"Failed to fetch PDF: {resp.status_code}")
    pdf_bytes = resp.content

    pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_count = len(pdf)

    blocks_out: List[Dict[str, Any]] = []
    page_texts: Dict[int, str] = {}

    for i in range(page_count):
        page_num = i + 1
        page = pdf[i]
        w, h = float(page.rect.width), float(page.rect.height)

        native_lines = extract_text_lines_with_bboxes(page)
        native_text = "\n".join([l["text"] for l in native_lines]).strip()

        use_ocr = len(native_text) < TEXT_MIN_CHARS_PER_PAGE
        lines = []
        if use_ocr:
            ocr_lines = ocr_page_lines(page, OCR_DPI)
            for idx, ln in enumerate(ocr_lines):
                x0,y0,x1,y1 = ln["bbox_abs"]
                lines.append({
                    "client_id": f"p{page_num}-l{idx}",
                    "page_number": page_num,
                    "block_type": "line",
                    "text": ln["text"],
                    "bbox_norm": bbox_norm_from_abs(x0,y0,x1,y1,w,h),
                    "confidence": ln.get("confidence"),
                    "meta": {"source": "ocr"},
                })
        else:
            for idx, ln in enumerate(native_lines):
                x0,y0,x1,y1 = ln["bbox_abs"]
                lines.append({
                    "client_id": f"p{page_num}-l{idx}",
                    "page_number": page_num,
                    "block_type": "line",
                    "text": ln["text"],
                    "bbox_norm": bbox_norm_from_abs(x0,y0,x1,y1,w,h),
                    "confidence": None,
                    "meta": {"source": "pdf_text"},
                })

        blocks_out.extend(lines)
        page_texts[page_num] = "\n".join([l["text"] for l in lines])

        if req.include_figures:
            figs = extract_figures(page)
            for fidx, f in enumerate(figs):
                x0,y0,x1,y1 = f["bbox_abs"]
                blocks_out.append({
                    "client_id": f"p{page_num}-f{fidx}",
                    "page_number": page_num,
                    "block_type": "figure",
                    "text": "Figure",
                    "bbox_norm": bbox_norm_from_abs(x0,y0,x1,y1,w,h),
                    "confidence": None,
                    "meta": {"source": "embedded_image"},
                })

    # Build chunks (+ optional embeddings)
    chunks = build_chunks_from_page_text(page_texts)
    if req.embed and chunks:
        embeddings = embed_texts([c["content"] for c in chunks])
        for c, e in zip(chunks, embeddings):
            c["embedding"] = e

    return {
        "doc_id": req.doc_id,
        "page_count": page_count,
        "blocks": blocks_out,
        "chunks": chunks,
    }

@app.post("/ask")
def ask(req: AskRequest, x_worker_key: Optional[str] = Header(default=None)):
    require_worker_key(x_worker_key)

    mode = req.mode
    question = (req.question or "").strip()
    if mode == "suggested_questions" and not question:
        question = "Generate 6 suggested questions a recruiter might ask about this document."

    mode_instructions = {
        "chat": "Answer the question.",
        "tldr": "Produce a TL;DR in 10 bullets.",
        "takeaways_risks": "Return (A) Key takeaways (5–8 bullets) and (B) Risks/limitations (5–8 bullets).",
        "metrics_table": "Extract key metrics/numbers into a compact table-like list (metric | value | context).",
        "glossary": "Extract key terms/jargon and define each in 1 sentence.",
        "contradictions": "List contradictions or ambiguities with evidence.",
        "suggested_questions": "Generate 6 tailored questions, each grounded in the doc."
    }
    instruction = mode_instructions.get(mode, mode_instructions["chat"])

    # Limit evidence for token safety
    evidence_blocks = req.evidence_blocks[:MAX_EVIDENCE_BLOCKS]
    block_by_id = {b.client_id: b for b in evidence_blocks}

    evidence_lines = []
    for b in evidence_blocks:
        t = (b.text or "").strip()
        if not t:
            continue
        evidence_lines.append(f"[block_id={b.client_id} p={b.page_number} type={b.block_type}] {t}")

    evidence_blob = "\n".join(evidence_lines)

    system = (
        "You are ProofRead, a verification-first document assistant.\n"
        "RULES:\n"
        "- Use ONLY the EVIDENCE blocks. No outside knowledge.\n"
        "- Every sentence in answer_text must cite 1+ block_id. If you can't support it, put that exact sentence in coverage.unsupported_sentences.\n"
        "- citations must reference valid block_id values from EVIDENCE.\n"
        "- Provide 3–6 evidence_cards, each with a claim and citation_ids.\n"
        "- coverage.grounded_pct should reflect how many answer sentences are supported.\n\n"
        "Return STRICT JSON with keys: answer_text, citations, evidence_cards, coverage, why_sources.\n"
        "citations[] must include: citation_id (block_id), snippet.\n"
        "evidence_cards[] must include: claim, citation_ids[].\n"
        "coverage must include: grounded_pct, unsupported_sentences[].\n"
        "why_sources must include: matched_terms[], top_pages[], ignored_sections[].\n"
    )

    top_pages = sorted(set([b.page_number for b in evidence_blocks]))[:10]
    user = (
        f"MODE: {mode}\n"
        f"INSTRUCTION: {instruction}\n"
        f"QUESTION: {question}\n"
        f"TOP_PAGES_USED: {top_pages}\n"
        f"IGNORED_SECTIONS: {req.ignored_sections}\n\n"
        f"EVIDENCE:\n{evidence_blob}\n"
    )

    raw = call_openai_json(system, user)

    # Canonicalize citations with bbox/page from evidence_blocks
    citations_out = []
    for c in (raw.get("citations") or []):
        cid = str(c.get("citation_id", "")).strip()
        b = block_by_id.get(cid)
        if not b:
            continue
        citations_out.append({
            "citation_id": cid,
            "page_number": b.page_number,
            "bbox_norm": b.bbox_norm,
            "snippet": (b.text or "")[:280],
            "block_type": b.block_type,
        })

    # Evidence cards + strength
    cards_out = []
    for ec in (raw.get("evidence_cards") or []):
        claim = (ec.get("claim") or "").strip()
        cids = [str(x) for x in (ec.get("citation_ids") or []) if str(x).strip()]
        if not claim or not cids:
            continue
        first = block_by_id.get(cids[0])
        snippet = (first.text if first else "") or ""
        cards_out.append({
            "claim": claim,
            "citation_ids": cids,
            "strength": compute_strength(claim, snippet),
        })

    answer_text = (raw.get("answer_text") or "").strip()
    unsupported = ((raw.get("coverage") or {}).get("unsupported_sentences") or [])
    sents = sentence_split(answer_text)
    grounded = sum(1 for s in sents if s not in unsupported)
    grounded_pct = int(round(100 * grounded / max(1, len(sents))))

    return {
        "answer_text": answer_text,
        "citations": citations_out,
        "evidence_cards": cards_out,
        "coverage": {
            "grounded_pct": grounded_pct,
            "unsupported_sentences": unsupported,
        },
        "why_sources": {
            "matched_terms": (raw.get("why_sources") or {}).get("matched_terms") or [],
            "top_pages": top_pages,
            "ignored_sections": (raw.get("why_sources") or {}).get("ignored_sections") or [],
        },
    }
