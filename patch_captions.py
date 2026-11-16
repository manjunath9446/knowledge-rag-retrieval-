import json
import re
import pymupdf as fitz
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PDF_PATH = SCRIPT_DIR / "pdfs" / "GT2026-178735.pdf"
CHUNKS_FILE = SCRIPT_DIR / "data" / "parsed_chunks.json"

FIGURE_PATTERN = re.compile(r"^\s*(fig(?:ure)?\.?\s*(\d+)[:.\-]?)", re.IGNORECASE)

def patch_captions():
    print("Patching image captions in JSON...")
    
    with open(CHUNKS_FILE, "r") as f:
        chunks = json.load(f)

    doc = fitz.open(PDF_PATH)
    
    # 1. Find the caption on every page
    page_captions = {}
    for page_num, page in enumerate(doc):
        text_blocks = page.get_text("blocks")
        for b in text_blocks:
            text = b[4].strip()
            m = FIGURE_PATTERN.match(text)
            if m:
                page_captions[page_num + 1] = {
                    "text": text,
                    "figure_number": int(m.group(2))
                }
                break # Just grab the first figure caption on the page

    # 2. Apply the caption to all images on that page
    patched_count = 0
    for chunk in chunks:
        if "image_path" in chunk:
            # Extract page number from path (e.g., "data\\images\\GT2026-178735_p4_img_6.png" -> 4)
            match = re.search(r"_p(\d+)_img_", chunk["image_path"])
            if match:
                page_num = int(match.group(1))
                
                if page_num in page_captions:
                    caption_data = page_captions[page_num]
                    chunk["text"] = f"Visual diagram. Figure {caption_data['figure_number']}. {caption_data['text']}"
                    chunk["figure_number"] = caption_data["figure_number"]
                    patched_count += 1
                    print(f"  -> Bound Page {page_num} image to: {caption_data['text']}")

    doc.close()

    with open(CHUNKS_FILE, "w") as f:
        json.dump(chunks, f, indent=2)
        
    print(f"✅ Patch complete! Bound {patched_count} images.")

if __name__ == "__main__":
    patch_captions()