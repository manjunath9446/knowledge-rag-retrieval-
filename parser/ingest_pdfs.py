import os
import json
import asyncio
import re
import sqlite3
import pandas as pd
from collections import defaultdict
from pathlib import Path
from io import StringIO
from llama_cloud import AsyncLlamaCloud
import pymupdf as fitz
from openai import OpenAI

LLAMA_CLOUD_API_KEY = os.environ.get("LLAMA_CLOUD_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not LLAMA_CLOUD_API_KEY or not GROQ_API_KEY:
    raise RuntimeError(
        "LLAMA_CLOUD_API_KEY and GROQ_API_KEY must both be set.\n"
        "For local dev: create .env at project root (see .env.example).\n"
        "For Streamlit Cloud: set them in the app's Secrets panel."
    )

groq_client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
PDF_FOLDER = PROJECT_ROOT / "pdfs"
IMAGE_FOLDER = PROJECT_ROOT / "data" / "images"
DB_FILE = PROJECT_ROOT / "data" / "hycosys_tables.db"
CHUNKS_FILE = PROJECT_ROOT / "data" / "parsed_chunks.json"

PDF_FOLDER.mkdir(exist_ok=True)
IMAGE_FOLDER.mkdir(parents=True, exist_ok=True)
DB_FILE.parent.mkdir(parents=True, exist_ok=True)

FIGURE_PATTERN = re.compile(
    r"^[\s\*_#>\-\u2022]*fig(?:ure)?\.?\s*(\d+)\b",
    re.IGNORECASE,
)

def extract_markdown_text(md_obj) -> str:
    if isinstance(md_obj, str):
        return md_obj
    if hasattr(md_obj, "pages"):
        return "\n\n".join([getattr(p, "markdown", getattr(p, "text", str(p))) for p in md_obj.pages])
    if hasattr(md_obj, "text"):
        return md_obj.text
    return str(md_obj)

def extract_images_and_captions(pdf_path, pdf_id):
    print(f"     [PyMuPDF] Extracting images & binding captions for {pdf_path.name}...")
    img_chunks = []
    fig_count = 0

    try:
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc):
            img_infos = page.get_image_info(xrefs=True)

            text_dict = page.get_text("dict")
            caption_candidates = []
            for block in text_dict.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    line_text = "".join(
                        span.get("text", "") for span in line.get("spans", [])
                    )
                    m = FIGURE_PATTERN.match(line_text)
                    if m:
                        lx0, ly0, lx1, ly1 = line["bbox"]
                        caption_candidates.append({
                            "bbox": (lx0, ly0, lx1, ly1),
                            "text": line_text.strip(),
                            "figure_number": int(m.group(1)),
                        })

            bboxes_by_figure = defaultdict(list)
            unbound_images = []

            for img in img_infos:
                ix0, iy0, ix1, iy1 = img["bbox"]
                best_caption, best_fig_num, best_gap = None, None, float("inf")
                for cand in caption_candidates:
                    cx0, cy0, cx1, cy1 = cand["bbox"]
                    vertical_gap = cy0 - iy1
                    horizontal_offset = abs(cx0 - ix0)
                    if 0 <= vertical_gap <= 40 and horizontal_offset <= 100:
                        if vertical_gap < best_gap:
                            best_caption, best_fig_num, best_gap = (
                                cand["text"], cand["figure_number"], vertical_gap
                            )

                if best_fig_num is not None:
                    bboxes_by_figure[best_fig_num].append({
                        "bbox": (ix0, iy0, ix1, iy1),
                        "caption": best_caption,
                    })
                else:
                    unbound_images.append({"bbox": (ix0, iy0, ix1, iy1)})

            for fig_num, entries in bboxes_by_figure.items():
                x0 = min(e["bbox"][0] for e in entries)
                y0 = min(e["bbox"][1] for e in entries)
                x1 = max(e["bbox"][2] for e in entries)
                y1 = max(e["bbox"][3] for e in entries)
                union_rect = fitz.Rect(x0, y0, x1, y1)

                raw_caption = next((e["caption"] for e in entries if e["caption"]), None)

                img_filename = f"{pdf_id}_p{page_num+1}_fig_{fig_num}.png"
                img_path = IMAGE_FOLDER / img_filename
                pix = page.get_pixmap(clip=union_rect, dpi=200)
                pix.save(str(img_path))

                if raw_caption:
                    cleaned = FIGURE_PATTERN.sub("", raw_caption, count=1).strip(" :.-")
                    chunk_text = (
                        f"Figure {fig_num}. {cleaned}" if cleaned else f"Figure {fig_num}."
                    )
                else:
                    chunk_text = f"Engineering diagram from page {page_num+1}."

                img_chunks.append({
                    "chunk_id": f"{pdf_id}_fig_{fig_num}",
                    "section_header": f"[Visual Diagram: {img_filename}]",
                    "text": chunk_text,
                    # FIX: Use posix path to ensure forward slashes for Linux
                    "image_path": img_path.relative_to(PROJECT_ROOT).as_posix(),
                    "figure_number": fig_num,
                    "doc_id": pdf_id,
                    "team": "structures",
                    "access_level": "engineering",
                    "revision": "A",
                    "approved": True,
                })
                fig_count += 1
                print(f"     -> Page {page_num+1}: merged {len(entries)} fragment(s) into Figure {fig_num}"
                      f"{f': {raw_caption}' if raw_caption else ' (unbound)'}")

            for idx, entry in enumerate(unbound_images):
                x0, y0, x1, y1 = entry["bbox"]
                img_filename = f"{pdf_id}_p{page_num+1}_unbound_{idx}.png"
                img_path = IMAGE_FOLDER / img_filename
                pix = page.get_pixmap(clip=fitz.Rect(x0, y0, x1, y1), dpi=200)
                pix.save(str(img_path))

                img_chunks.append({
                    "chunk_id": f"{pdf_id}_p{page_num+1}_unbound_{idx}",
                    "section_header": f"[Visual Diagram: {img_filename}]",
                    "text": f"Engineering diagram from page {page_num+1}. (caption not found -- flagged for QA)",
                    # FIX: Use posix path to ensure forward slashes for Linux
                    "image_path": img_path.relative_to(PROJECT_ROOT).as_posix(),
                    "figure_number": None,
                    "doc_id": pdf_id,
                    "team": "structures",
                    "access_level": "engineering",
                    "revision": "A",
                    "approved": True,
                    "needs_qa_review": True,
                })

        doc.close()
    except Exception as e:
        print(f"     [Image Extraction Error] {e}")

    print(f"     [PyMuPDF] Extracted {fig_count} bound figures "
          f"({len(img_chunks) - fig_count} unbound, flagged for QA).")
    return img_chunks

async def parse_pdf(client, pdf_path, existing_chunks, conn):
    print(f"  -> Parsing {pdf_path.name} with LlamaParse...")
    pdf_id = pdf_path.stem.upper()

    image_chunks = extract_images_and_captions(pdf_path, pdf_id)
    existing_chunks.extend(image_chunks)

    try:
        with open(pdf_path, "rb") as f:
            file_obj = await client.files.create(file=f, purpose="parse")

        job = await client.parsing.parse(
            file_id=file_obj.id, tier="agentic", version="latest", expand=["markdown"]
        )

        while job.job.status not in ("COMPLETED", "FAILED", "CANCELLED"):
            await asyncio.sleep(3)
            job = await client.parsing.get(job.job.id, expand=["markdown"])

        if job.job.status == "FAILED":
            print(f"     [ERROR] LlamaParse job failed.")
            return existing_chunks

        full_markdown = extract_markdown_text(job.markdown)

        html_tables = re.findall(r"(<table.*?</table>)", full_markdown, re.DOTALL | re.IGNORECASE)
        md_tables = re.findall(r"((?:^[ \t]*\|.*\|[ \t]*$\n?)+)", full_markdown, re.MULTILINE)
        all_tables = html_tables + md_tables

        table_count = 0
        for idx, table_data in enumerate(all_tables):
            try:
                if table_data.strip().startswith("<table"):
                    df = pd.read_html(StringIO(table_data))[0]
                else:
                    df = pd.read_csv(StringIO(table_data), sep="|", skipinitialspace=True)
                    df = df.dropna(axis=1, how="all")
                    df.columns = [c.strip() for c in df.columns if c.strip() != ""]

                if not df.empty and len(df.columns) > 1:
                    table_name = f"{pdf_id}_tbl_{table_count+1}"
                    df["doc_id"] = pdf_id
                    df.to_sql(table_name, conn, if_exists="replace", index=False)
                    
                    # Convert table to Markdown format for perfect UI rendering
                    drop_cols = ["doc_id"] if "doc_id" in df.columns else []
                    df_clean = df.drop(columns=drop_cols).fillna("")
                    
                    headers = " | ".join(str(col) for col in df_clean.columns)
                    separator = " | ".join(["---"] * len(df_clean.columns))
                    rows = [" | ".join(str(val) for val in row) for _, row in df_clean.iterrows()]
                    
                    md_table = f"**Table Name:** `{table_name}`\n\n| {headers} |\n| {separator} |\n" + "\n".join(f"| {row} |" for row in rows)
                    
                    existing_chunks.append({
                        "chunk_id": f"{table_name}_text_chunk",
                        "section_header": f"[Table Data: {table_name}]",
                        "text": md_table,
                        "doc_id": pdf_id,
                        "team": "structures",
                        "access_level": "engineering",
                        "revision": "A",
                        "approved": True,
                    })
                        
                    table_count += 1
            except Exception:
                pass

        sections = re.split(r"\n(?=#{1,3} )", full_markdown)
        for sec in sections:
            if not sec.strip():
                continue
            header_match = re.match(r"^(#{1,3} .*)", sec)
            header = header_match.group(1) if header_match else "General"

            clean_text = sec
            clean_text = re.sub(r"<table.*?</table>", " ", clean_text, flags=re.DOTALL | re.IGNORECASE)
            clean_text = re.sub(r"(?:^[ \t]*\|.*\|[ \t]*$\n?)+", " ", clean_text, flags=re.MULTILINE)
            clean_text = re.sub(
                r"^[ \t\*_#>]*\**\s*(?:figure|fig\.?)\s*\d+\s*[:.]\**\s*[^\n]*$",
                "", clean_text, flags=re.MULTILINE | re.IGNORECASE,
            )
            clean_text = re.sub(
                r"^[ \t\*_#>]*(?:drawing\s*:|engineering\s+drawing|line\s+chart|"
                r"image\s+of|photograph\s+of|diagram\s+of|plot\s+of|graph\s+of)[^\n]*$",
                "", clean_text, flags=re.MULTILINE | re.IGNORECASE,
            )
            clean_text = re.sub(r"^[ \t]*\d{1,3}[ \t]*$", "", clean_text, flags=re.MULTILINE)
            clean_text = re.sub(r"\n{3,}", "\n\n", clean_text).strip()
            if not clean_text:
                continue

            existing_chunks.append({
                "chunk_id": f"{pdf_id}_chunk_{len(existing_chunks)}",
                "section_header": header,
                "text": clean_text,
                "doc_id": pdf_id,
                "team": "structures",
                "access_level": "engineering",
                "revision": "A",
                "approved": True,
            })

        print(f"     [SUCCESS] {pdf_path.name}: {table_count} tables, {len(image_chunks)} figure chunks, text processed.")

    except Exception as e:
        print(f"     [FATAL ERROR] {e}")

    return existing_chunks

async def ingest_files(pdf_paths):
    client = AsyncLlamaCloud(
        api_key=LLAMA_CLOUD_API_KEY,
        base_url="https://api.cloud.llamaindex.ai",
    )

    existing_chunks = []
    if CHUNKS_FILE.exists():
        with open(CHUNKS_FILE, "r") as f:
            existing_chunks = json.load(f)

    conn = sqlite3.connect(DB_FILE)
    processed = 0
    try:
        for pdf in pdf_paths:
            pdf = Path(pdf)
            if not pdf.exists():
                print(f"  [SKIP] {pdf} does not exist.")
                continue
            existing_chunks = await parse_pdf(client, pdf, existing_chunks, conn)
            processed += 1
    finally:
        conn.close()

    with open(CHUNKS_FILE, "w") as f:
        json.dump(existing_chunks, f, indent=2)

    return processed, len(existing_chunks)

async def main():
    pdfs = list(PDF_FOLDER.glob("*.pdf"))
    n, total = await ingest_files(pdfs)
    print(f"\n[SUCCESS] Ingested {n} PDF(s). Total chunks in database: {total}")

if __name__ == "__main__":
    asyncio.run(main())