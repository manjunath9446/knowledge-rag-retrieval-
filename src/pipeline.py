import os
import re
import json
import sqlite3
import pandas as pd
import base64
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

from access.rules import RuleACL, UserContext

LLAMA_CLOUD_API_KEY = os.environ.get("LLAMA_CLOUD_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not LLAMA_CLOUD_API_KEY or not GROQ_API_KEY:
    raise RuntimeError(
        "LLAMA_CLOUD_API_KEY and GROQ_API_KEY must both be set.\n"
        "For local dev: create .env at project root (see .env.example).\n"
        "For Streamlit Cloud: set them in the app's Secrets panel."
    )

groq_client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

SPAN_SELECTOR_MODEL = "openai/gpt-oss-20b"
SPAN_SELECTOR_FALLBACK_MODEL = "qwen/qwen3.8-27b"

_SQL_FORBIDDEN = re.compile(
    r"\b(DROP|DELETE|INSERT|UPDATE|ATTACH|DETACH|ALTER|PRAGMA|UNION|EXEC|REPLACE)\b",
    re.IGNORECASE,
)

_MATH_LINE = re.compile(
    r"\\\(|\\\[|\$\$|"
    r"\\[a-zA-Z]{2,}|"
    r"[A-Za-z]\s*_\s*\{|"
    r"[A-Za-z]\s*\^\s*\{|"
    r"\([0-9]+\)\s*$", 
)

_CAPTION_START = re.compile(
    r"^\s*\**\s*(?:figure|fig\.?)\s*\d+|"
    r"^\s*(?:drawing\s*:|engineering\s+drawing|line\s+chart|"
    r"image\s+of|photograph\s+of|diagram\s+of|plot\s+of|graph\s+of)",
    re.IGNORECASE,
)

_LEFT_DELIM = r'\\left\\?[\(\[\{\|\.]'
_RIGHT_DELIM = r'\\right\\?[\)\]\}\|\.]'

_GENERIC_VISUAL = {
    "diagram", "figure", "fig", "chart", "graph", "plot", "image",
    "picture", "look", "show", "see", "given", "what", "which",
    "who", "when", "where", "why", "how", "the", "a", "an", "is",
    "are", "was", "were", "of", "in", "on", "at", "to", "for",
    "and", "or", "this", "that", "it", "does", "do", "can",
    "could", "would", "please", "explain",
}

_PROSE_EXTRA_STOPWORDS = {"campbell"}

_SPAN_STOP = {
    "what", "which", "who", "when", "where", "why", "how",
    "the", "a", "an", "is", "are", "was", "were", "of", "in",
    "on", "at", "to", "for", "and", "or", "this", "that", "it",
    "does", "do", "can", "could", "would", "should", "please",
    "tell", "show", "give", "find", "get", "see", "look",
    "value", "values", "actually", "supposed", "measure", "measured",
    "represent",
}

_SEED_SYNONYMS = {
    "hardness":    {"hardness", "hb", "brinell"},
    "temperature": {"temperature", "°c", "deg", "celsius"},
    "speed":       {"speed", "rpm", "hz"},
    "stress":      {"stress", "mpa", "gpa", "psi"},
}

SPAN_SELECTOR_PROMPT = """You are a sentence selector. Given numbered sentences from an engineering report, select the sentences that best answer the question.

═══ SENTENCES ═══
{sentence_block}
════════════════

QUESTION: {query}

RULES:
- Return ONLY a comma-separated list of sentence IDs (e.g., S1, S4).
- Do NOT output JSON. Do NOT use brackets. Do NOT include prose.
- If no sentence answers the question, return exactly: NONE
- Select 2-4 sentences maximum. Prefer shorter, more specific sentences.

TABLE QUESTIONS (CRITICAL):
- If the question asks for a table, matrix, or data rows, you MUST select the rows containing the data.
- Look for sentences that start with "Table Name:" or contain "|" separators (e.g., "Test Case: 1 | Shape: Pill").
- DO NOT just return the table caption (e.g., "Table 2: ..."). You must return the actual data.

MULTI-PART QUESTIONS:
- If the question asks "X or Y", "X vs Y", "was it X or Y", or otherwise
  requires TWO OR MORE factors, select at least one sentence for EACH factor.
- Look at the §SECTION labels: if two different sections could each answer
  part of the question, pick from both.

EQUATION / FORMULA QUESTIONS (CRITICAL):
- If the question references an equation by number ("equation 3", "Eq. 5"),
  you MUST select BOTH:
    1) the math line containing the equation itself.
    2) the prose sentence describing it.
- NEVER select the Nomenclature list ([M] ; Mass Matrix) when asked for an equation.

SINGLE-PART QUESTIONS:
- Select 1-3 sentences from the most relevant section.

OUTPUT (comma-separated IDs or NONE):"""


class EngineeringRAG:
    _NO_ANSWER_SENTINEL = "__NO_ANSWER_IN_REPORT__"

    def __init__(self):
        self.acl = RuleACL()

        self.chunks = []
        chunks_file = Path("data/parsed_chunks.json")
        if chunks_file.exists():
            with open(chunks_file, "r") as f:
                self.chunks = json.load(f)
        if not self.chunks:
            raise ValueError("No parsed chunks found. Run 'python parser/ingest_pdfs.py' first!")

        for c in self.chunks:
            c["contextual_text"] = f"{c['section_header']}. {c['text']}"
            c["component"] = "general"

        self.bm25 = BM25Okapi([re.findall(r"\w+", c["contextual_text"].lower()) for c in self.chunks])

        print("[INFO] Loading BGE Embedding Model...")
        self.embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
        self.dense_matrix = self.embedder.encode(
            [c["contextual_text"] for c in self.chunks], show_progress_bar=False
        )

        self.db_path = Path("data/hycosys_tables.db")
        if self.db_path.exists():
            self.db_conn = sqlite3.connect(
                f"file:{self.db_path}?mode=ro", uri=True, check_same_thread=False,
            )
            self.table_schemas = self._get_schemas()
        else:
            self.db_conn = None
            self.table_schemas = {}

        self.synonym_map: dict[str, set[str]] = {k: set(v) for k, v in _SEED_SYNONYMS.items()}
        auto = self._parse_nomenclature(self.chunks)
        for k, v in auto.items():
            self.synonym_map.setdefault(k, set()).update(v)
        if auto:
            print(f"[INFO] Auto-built {len(auto)} synonym pairs from NOMENCLATURE")

        self._sentence_cache = {}

    def _get_schemas(self):
        cursor = self.db_conn.cursor()
        cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table';")
        return {row[0]: row[1] for row in cursor.fetchall()}

    @staticmethod
    def _parse_nomenclature(chunks: list) -> dict[str, set[str]]:
        pairs: dict[str, set[str]] = {}
        pattern = re.compile(
            r"^\s*([A-Za-z][A-Za-z0-9_]{1,10})\s*[:=\-]\s*(.+?)\s*$",
            re.MULTILINE,
        )
        for c in chunks:
            if "image_path" in c:
                continue
            header = (c.get("section_header") or "").lower()
            if "nomenclature" not in header and "abbreviation" not in header:
                continue
            for m in pattern.finditer(c.get("text") or ""):
                abbrev = m.group(1).strip().lower()
                expansion = m.group(2).strip().lower()
                if len(abbrev) < 2 or len(expansion) < 3:
                    continue
                if abbrev in {"fig", "eq", "see", "table", "the", "for"}:
                    continue
                expansion_words = [w for w in expansion.split() if len(w) >= 2]
                pairs.setdefault(abbrev, set()).update({abbrev, expansion, *expansion_words})
                for w in expansion_words[:2]:
                    pairs.setdefault(w, set()).update({abbrev, w})
        return pairs

    @staticmethod
    def _clean_header(header: str) -> str:
        if not header:
            return ""
        h = re.sub(r"^\[Visual Diagram:[^\]]*\]\s*", "", header)
        h = re.sub(r"^\[Table Data:[^\]]*\]\s*", "", header)
        h = re.sub(r"^[#>\s]+", "", h)
        h = h.replace("**", "").replace("__", "")
        return h.strip(" :-") or "General"

    @staticmethod
    def _format_df(df: pd.DataFrame) -> str:
        df = df.copy()
        df = df.replace(["None", "nan", "NaN", "null", "NULL", ""], np.nan)
        df = df.dropna(axis=1, how="all")
        if "doc_id" in df.columns:
            df = df.drop(columns=["doc_id"])
        keep = [c for c in df.columns if df[c].notna().any()]
        df = df[keep].fillna("")
        if df.empty or df.shape[1] == 0:
            return "(no populated columns in this table)"
        return df.to_string(index=False)

    @staticmethod
    def _sql_result_relevant(query: str, df: pd.DataFrame) -> bool:
        if df.empty or df.shape[1] == 0:
            return False
        _IGNORE_COLS = {"doc_id", "chunk_id", "team", "access_level"}
        searchable_cols = [c for c in df.columns if c.lower() not in _IGNORE_COLS]
        if not searchable_cols:
            return False
        terms = [w for w in re.findall(r"\w+", query.lower()) if len(w) >= 2 and w not in _SPAN_STOP]
        if not terms:
            return True
        haystack = " ".join(str(c) for c in searchable_cols).lower()
        for col in searchable_cols:
            haystack += " " + " ".join(df[col].astype(str)).lower()
        return any(re.search(rf"\b{re.escape(t)}\b", haystack) for t in terms)

    @staticmethod
    def _clean_vqa_answer(text: str) -> tuple[str, bool]:
        is_unknown = "UNKNOWN" in text.upper()
        analysis = re.sub(
            r"\n*\s*UNKNOWN\s*[-—].*$", "", text,
            flags=re.IGNORECASE | re.MULTILINE,
        ).strip()
        return (analysis or text.strip(), is_unknown)

    @staticmethod
    def _split_into_units(text: str) -> list[str]:
        if text.startswith("Table Name:") or "[Table Data:" in text:
            units = []
            for line in text.split("\n"):
                line = line.strip()
                if line:
                    units.append(line)
            return units

        units = []
        math_spans = []
        for pattern in (r"\$\$(.+?)\$\$", r"\\\[(.+?)\\\]", r"\\\((.+?)\\\)"):
            for m in re.finditer(pattern, text, re.DOTALL):
                math_spans.append((m.start(), m.end(), m.group(0)))
        math_spans.sort()

        cursor = 0
        for start, end, math_content in math_spans:
            prose = text[cursor:start]
            if prose.strip():
                for sent in re.split(r"(?<=[.!?])\s+", prose):
                    sent = sent.strip()
                    if sent:
                        units.append(sent)
            units.append(math_content.strip())
            cursor = end

        remaining = text[cursor:]
        if remaining.strip():
            for sent in re.split(r"(?<=[.!?])\s+", remaining):
                sent = sent.strip()
                if sent:
                    units.append(sent)

        merged_units = []
        for u in units:
            u = u.strip()
            if not u:
                continue
            if merged_units and (re.match(r'^[\^_]\s*\{', u) or re.fullmatch(r'\(\d+\)', u)):
                merged_units[-1] = merged_units[-1] + " " + u
            else:
                merged_units.append(u)

        return merged_units

    @staticmethod
    def _fix_math_markup(text: str) -> str:
        if not text:
            return text

        lines = text.split('\n')
        if len(lines) > 5 and (sum(len(l.strip()) for l in lines) / len(lines)) < 3:
            text = text.replace('\n', ' ')
            text = re.sub(r'\s+', ' ', text)

        text = re.sub(r'\\begin\s*\{\s*aligned\s*\}', r'\\begin{aligned}', text)
        text = re.sub(r'\\end\s*\{\s*aligned\s*\}', r'\\end{aligned}', text)

        def extract_aligned(match):
            core = match.group(1)
            eq_num = match.group(2)

            num_left = len(re.findall(_LEFT_DELIM, core))
            num_right = len(re.findall(_RIGHT_DELIM, core))
            if num_right > num_left:
                excess = num_right - num_left
                for m in list(re.finditer(_RIGHT_DELIM, core))[-excess:]:
                    core = core[:m.start()] + m.group(0)[-1] + core[m.end():]
            elif num_left > num_right:
                excess = num_left - num_right
                for m in list(re.finditer(_LEFT_DELIM, core))[:excess]:
                    core = core[:m.start()] + m.group(0)[-1] + core[m.end():]

            tail = f" \\quad ({eq_num})" if eq_num else ""
            return f"\n\n$$\n{core.strip()}{tail}\n$$\n\n"

        aligned_pattern = (
            r'(?:\$\$\s*)?'
            r'((?:' + _LEFT_DELIM + r'\s*)?'
            r'\\begin\{aligned\}.*?\\end\{aligned\}'
            r'(?:\s*' + _RIGHT_DELIM + r')?)'
            r'\s*(?:\$\$)?'
            r'\s*(?:\\quad\s*)?\s*(?:\((\d+)\))?'
        )
        text = re.sub(aligned_pattern, extract_aligned, text, flags=re.DOTALL)

        def fold_eq_num(match):
            core = match.group(1)[2:-2].strip()
            num = match.group(2)
            if num and f"({num})" not in core:
                core = f"{core} \\quad ({num})"
            return f"\n\n$$\n{core}\n$$\n\n"

        text = re.sub(
            r'(\$\$.*?\$\$)\s*(?:\\quad\s*)?\s*(?:\((\d+)\))?',
            fold_eq_num, text, flags=re.DOTALL,
        )

        single_pattern = re.compile(
            r'([^.!?]+?(?:\\Omega|\\sum|\\frac|\\sqrt|\\cdot|\{|=)[^.!?]+?)\s*(?:\((\d+)\))'
        )
        def extract_single(match):
            tail = f" \\quad ({match.group(2)})" if match.group(2) else ""
            return f"\n\n$$\n{match.group(1).strip()}{tail}\n$$\n\n"

        parts = re.split(r'(\$\$.*?\$\$)', text, flags=re.DOTALL)
        for i in range(len(parts)):
            if not parts[i].startswith('$$'):
                parts[i] = single_pattern.sub(extract_single, parts[i])
        text = "".join(parts)

        parts = re.split(r'(\$\$.*?\$\$)', text, flags=re.DOTALL)
        for i in range(len(parts)):
            if parts[i].startswith('$$'):
                parts[i] = parts[i].replace('$$$$', '$$')
        text = "".join(parts)
        return re.sub(r'\n{3,}', '\n\n', text).strip()

    def _answer_with_image(self, image_path: str, query: str) -> str:
        try:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
            prompt = f"""
            You are a senior aerospace engineer. Look at this technical diagram.
            Answer the user's question based ONLY on what is EXPLICITLY LABELED in the image.

            CRITICAL RULE: If a specific number, value, or data point is asked for, and it is NOT explicitly written on the chart, reply exactly: "UNKNOWN - The value is not explicitly labeled on this diagram." DO NOT guess.

            Question: {query}
            """
            response = groq_client.chat.completions.create(
                model="qwen/qwen3.8-27b",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                }],
                temperature=0,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Visual analysis failed: {e}"

    def _get_doc_sentences(self, doc_id: str):
        if doc_id in self._sentence_cache:
            return self._sentence_cache[doc_id]

        candidates = []
        for c in self.chunks:
            if c.get("doc_id") != doc_id or "image_path" in c:
                continue
            text = c["text"]
            text = re.sub(r"<table.*?</table>", " ", text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"(?:^[ \t]*\|.*\|[ \t]*$\n?)+", " ", text, flags=re.MULTILINE)
            for unit in self._split_into_units(text):
                is_table_line = " | " in unit or unit.startswith("Table Name:")
                if len(unit) < 25 and not is_table_line:
                    continue
                if _CAPTION_START.match(unit):
                    continue
                candidates.append({"text": unit, "header": c.get("section_header", "")})

        if candidates:
            vecs = self.embedder.encode([c["text"] for c in candidates], show_progress_bar=False)
            for c, v in zip(candidates, vecs):
                c["vec"] = v

        self._sentence_cache[doc_id] = candidates
        return candidates

    def _lookup_supporting_data(self, query: str, doc_id: str, top_k: int = 3) -> str | None:
        sections = []

        if self.db_conn:
            keywords = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 3]
            if keywords:
                doc_tables = [name for name in self.table_schemas if name.startswith(doc_id)]
                table_blocks = []
                for table in doc_tables:
                    try:
                        df = pd.read_sql_query(f'SELECT * FROM "{table}"', self.db_conn)
                    except Exception:
                        continue
                    mask = pd.Series(False, index=df.index)
                    for col in df.select_dtypes(include="object").columns:
                        col_lower = df[col].astype(str).str.lower()
                        for kw in keywords:
                            mask |= col_lower.str.contains(kw, na=False)
                    hit_rows = df[mask]
                    if not hit_rows.empty:
                        rendered = self._format_df(hit_rows)
                        if rendered and "no populated columns" not in rendered:
                            table_blocks.append(f"**`{table}`**\n```\n{rendered}\n```")
                if table_blocks:
                    sections.append("**📊 From report tables:**\n\n" + "\n\n".join(table_blocks))

        candidates = self._get_doc_sentences(doc_id)
        if candidates:
            prose_stopwords = _GENERIC_VISUAL | _PROSE_EXTRA_STOPWORDS
            query_words = [w for w in re.findall(r"\w+", query.lower())
                           if len(w) > 3 and w not in prose_stopwords]
            q_vec = self.embedder.encode([query])[0]
            sims = cosine_similarity([q_vec], [c["vec"] for c in candidates])[0]
            scored = []
            for c, s in zip(candidates, sims):
                score = float(s)
                if re.search(r"\d+(?:[.,]\d+)?\s*(?:rpm|hz|khz|%|mm|n\b|mpa|gpa|°c|deg)",
                             c["text"], re.IGNORECASE):
                    score += 0.20
                elif re.search(r"\b\d{2,}\b", c["text"]):
                    score += 0.08
                if query_words:
                    sent_lower = c["text"].lower()
                    present = sum(1 for w in query_words if w in sent_lower)
                    if present >= 2:
                        score += 0.10 * min(present, 3)
                scored.append((score, c))

            scored.sort(key=lambda x: -x[0])
            top = [s for s in scored[:top_k] if s[0] >= 0.40]
            if top:
                lines = "\n".join(f"- *{self._clean_header(c['header'])}* — {c['text']}"
                                  for _, c in top)
                sections.append("**📄 From report text:**\n\n" + lines)

        return "\n\n".join(sections) if sections else None

    def _call_selector(self, model: str, prompt: str) -> list[str]:
        response = groq_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_completion_tokens=800,
            extra_body={"reasoning_effort": "low"},
        )
        raw = response.choices[0].message.content.strip()
        
        if "NONE" in raw.upper():
            return []
            
        ids = re.findall(r"\bS\d+\b", raw)
        return ids

    def _select_and_assemble(self, query: str, doc_chunks: list) -> tuple[str | None, list[str]]:
        if not doc_chunks:
            return None, []

        sentence_map: dict[str, tuple[str, str]] = {}
        sentence_lines: list[str] = []
        for ci, c in enumerate(doc_chunks[:3], 1):
            text = (c.get("text") or "").strip()
            if not text:
                continue
            chunk_id = c.get("chunk_id") or c.get("doc_id") or f"E{ci}"
            section = self._clean_header(c.get("section_header", ""))
            for unit in self._split_into_units(text):
                unit = unit.strip()
                if not unit:
                    continue
                if _CAPTION_START.match(unit):
                    continue
                is_math = bool(_MATH_LINE.search(unit))
                is_table_line = " | " in unit or unit.startswith("Table Name:")
                if not is_math and not is_table_line and len(unit) < 25:
                    continue
                
                sid = f"S{len(sentence_map) + 1}"
                sentence_map[sid] = (chunk_id, unit)
                sentence_lines.append(f"[{sid} §{section}] {unit}")

        if not sentence_map:
            return None, []

        prompt = SPAN_SELECTOR_PROMPT.format(
            sentence_block="\n".join(sentence_lines),
            query=query,
        )

        selected_ids: list[str] = []
        try:
            selected_ids = self._call_selector(SPAN_SELECTOR_MODEL, prompt)
        except Exception as e:
            print(f"[SELECT] Primary ({SPAN_SELECTOR_MODEL}) failed: {e}")

        if not selected_ids:
            print(f"[SELECT] Primary returned no IDs — trying {SPAN_SELECTOR_FALLBACK_MODEL}")
            try:
                selected_ids = self._call_selector(SPAN_SELECTOR_FALLBACK_MODEL, prompt)
            except Exception as e:
                print(f"[SELECT] Fallback selector failed: {e}")

        selected = []
        for sid in selected_ids:
            sid = str(sid).strip()
            if sid in sentence_map:
                selected.append(sid)
            else:
                print(f"[SELECT] Unknown id {sid!r} — dropped")

        if not selected:
            return self._NO_ANSWER_SENTINEL, []

        query_terms = {w for w in re.findall(r"\w+", query.lower())
                       if len(w) > 3 and w not in _SPAN_STOP}
        expanded_terms = set(query_terms)
        for qt in list(query_terms):
            if qt in self.synonym_map:
                expanded_terms |= self.synonym_map[qt]

        query_numbers = set(re.findall(r"\b\d+\b", query))

        def _is_relevant(sid: str) -> bool:
            text = sentence_map[sid][1]
            text_lower = text.lower()
            if _MATH_LINE.search(text):
                return True
            if any(f"({n})" in text or f"equation {n}" in text_lower for n in query_numbers):
                return True
            return any(t in text_lower for t in expanded_terms)

        if not any(_is_relevant(sid) for sid in selected):
            print(f"[SELECT] Relevance gate: ALL {len(selected)} sentences rejected")
            return self._NO_ANSWER_SENTINEL, []

        kept = selected
        rejected = [sid for sid in selected if not _is_relevant(sid)]
        if rejected:
            print(f"[SELECT] Relevance gate: kept {len(kept)}, trusted model's {rejected}")

        num_pattern = re.compile(r"(\d+(?:[.,]\d+)?)\s*(HB|rpm|Hz|MPa|GPa|%|mm|°C|deg)",
                                 re.IGNORECASE)
        by_unit: dict[str, dict[str, set[str]]] = {}
        for sid in kept:
            for m in num_pattern.finditer(sentence_map[sid][1]):
                unit = m.group(2).lower()
                value = m.group(1)
                by_unit.setdefault(unit, {}).setdefault(value, set()).add(sid)

        conflicts: dict[str, dict[str, set[str]]] = {}
        for unit, value_sids in by_unit.items():
            if len(value_sids) < 2:
                continue
            mentions = [(v, sid) for v, sids in value_sids.items() for sid in sids]
            if any(
                v1 != v2 and s1 != s2
                for i, (v1, s1) in enumerate(mentions)
                for (v2, s2) in mentions[i + 1:]
            ):
                conflicts[unit] = value_sids

        if conflicts:
            lines = []
            conflict_sids = set()
            for unit, value_sids in conflicts.items():
                for value, sids in value_sids.items():
                    for sid in sids:
                        lines.append(f"- **{value} {unit}** — {sentence_map[sid][1]}")
                        conflict_sids.add(sid)
            for sid in kept:
                if sid not in conflict_sids:
                    lines.append(f"- {sentence_map[sid][1]}")
            body = "The report states different values for the same quantity:\n\n" + "\n".join(lines)
        else:
            body = " ".join(sentence_map[sid][1] for sid in kept)

        answer_nums = set(re.findall(r"\d+(?:[.,]\d+)?", body))
        source_nums = set()
        for sid in kept:
            source_nums.update(re.findall(r"\d+(?:[.,]\d+)?", sentence_map[sid][1]))
        untraceable = answer_nums - source_nums
        if untraceable:
            print(f"[SELECT] Numeric killswitch: {untraceable} not in sources")
            return self._NO_ANSWER_SENTINEL, []

        print(f"\n[DEBUG RAW BODY]:\n{body!r}\n")
        body = self._fix_math_markup(body)
        print(f"[DEBUG FIXED BODY]:\n{body!r}\n")
        
        used_chunk_ids = list({sentence_map[sid][0] for sid in kept})
        return body, used_chunk_ids

    def answer(self, query: str, user: UserContext, threshold=0.50):
        query = (query or "").strip()
        if not query or len(query) < 2 or not re.search(r"[a-zA-Z]", query):
            return {"answer": "Please ask a specific question about the engineering documentation.",
                    "refused": True, "sources": [], "confidence": 0.0}

        if not getattr(user, "teams", frozenset()) and user.clearance != "leadership":
            return {"answer": "Access denied.", "refused": True, "sources": [], "confidence": 0.0}

        restricted_keywords = ["compensation", "salary", "budget", "financial", "cost"]
        if user.clearance != "leadership" and any(w in query.lower() for w in restricted_keywords):
            return {"answer": "Access denied: Financial data is restricted.",
                    "refused": True, "sources": [], "confidence": 0.0}

        allowed = [i for i, c in enumerate(self.chunks) if self.acl.visible(c, user)]
        if not allowed:
            return {"answer": "Access denied.", "refused": True, "sources": [], "confidence": 0.0}

        # --- DETERMINISTIC EQUATION LOOKUP ---
        eq_match = re.search(r'\b(?:equation|eq\.?)\s*(\d+)\b', query, re.IGNORECASE)
        if eq_match:
            target_eq = eq_match.group(1)
            print(f"[EXACT MATCH] Query references Equation {target_eq}. Bypassing vector search.")
            
            exact_chunks = []
            for i in allowed:
                text = self.chunks[i]["text"]
                if re.search(rf'\({target_eq}\)|(?:Equation|Eq\.?)\s*{target_eq}\b', text, re.IGNORECASE):
                    exact_chunks.append(self.chunks[i])
            
            if exact_chunks:
                result, used_chunk_ids = self._select_and_assemble(query, exact_chunks[:3])
                if result and result != self._NO_ANSWER_SENTINEL:
                    return {
                        "answer": result,
                        "refused": False,
                        "sources": [{"doc_id": r["doc_id"], "title": r.get("section_header", "Unknown")}
                                    for r in exact_chunks],
                        "confidence": 1.0
                    }
            else:
                print(f"[EXACT MATCH] No chunk found containing ({target_eq})")

        figure_ref_regex = re.compile(r'figure\s*(\d+)|fig\.?\s*(\d+)', re.IGNORECASE)
        fig_match = figure_ref_regex.search(query)
        if fig_match:
            fig_num = int(fig_match.group(1) or fig_match.group(2))
            print(f"[EXACT MATCH] Figure {fig_num} — bypassing vector search")
            exact_images = [self.chunks[i] for i in allowed
                            if self.chunks[i].get("figure_number") == fig_num
                            and "image_path" in self.chunks[i]]
            if exact_images:
                img_chunk = exact_images[0]
                print(f"[VQA] Analyzing image: {img_chunk['image_path']}")
                raw_answer = self._answer_with_image(img_chunk["image_path"], query)
                analysis, is_unknown = self._clean_vqa_answer(raw_answer)
                body_parts = [analysis]
                sources = [{
                    "doc_id": img_chunk["doc_id"],
                    "title": img_chunk.get("text", img_chunk.get("section_header", "Unknown")),
                    "image_path": img_chunk["image_path"],
                }]
                if is_unknown:
                    body_parts.append("⚠️ *The value is not explicitly labeled on this diagram.*")
                    table_answer = self._lookup_supporting_data(query, img_chunk["doc_id"])
                    if table_answer:
                        body_parts.append("---")
                        body_parts.append("**Supporting evidence found in the report:**")
                        body_parts.append(table_answer)
                        sources.append({"doc_id": img_chunk["doc_id"], "title": "Tables + Text"})
                return {"answer": f"**Visual Analysis (Figure {fig_num}):**\n\n" + "\n\n".join(body_parts),
                        "refused": False, "sources": sources, "confidence": 1.0}

        visual_keywords = ["diagram", "figure", "chart", "graph", "plot", "image", "picture"]
        is_visual_query = any(w in query.lower() for w in visual_keywords)
        
        # NEW: Table Query Intent Detection
        is_table_query = any(w in query.lower() for w in ["table", "grid", "matrix", "rows", "columns", "spreadsheet", "tabulate"])

        if is_visual_query:
            print("[INTENT] Visual query detected — filtering to images only")
            allowed = [i for i in allowed if "image_path" in self.chunks[i]]
            if not allowed:
                return {"answer": "No images found in accessible documents.",
                        "refused": True, "sources": [], "confidence": 0.0}
            route = "VECTOR"
        else:
            allowed_doc_ids = list(set(self.chunks[i]["doc_id"] for i in allowed))
            route = "VECTOR"

            if self.db_conn:
                schemas_text = "\n".join(self.table_schemas.values())
                prompt = f"""
                You are an aerospace data router. Schemas: {schemas_text}

                CRITICAL RULES:
                1. Table names contain hyphens. Wrap table names in double quotes.
                2. Only route SQL when the question asks about a value that
                   EXISTS as a column in one of the tables. If the topic is not
                   in any schema (hardness, material properties, causes,
                   equations), route VECTOR.
                3. If no column is clearly relevant, do NOT force a SQL query.

                EXAMPLES:
                User: "What is A1?" -> {{"route": "SQL", "sql_query": "SELECT * FROM \\"GT2026-178735_tbl_1\\" WHERE \\"#No\\" = 'A1'"}}
                User: "Why did the shaft fail?" -> {{"route": "VECTOR"}}
                User: "What hardness did we measure?" -> {{"route": "VECTOR"}}
                User: "Explain equation 3" -> {{"route": "VECTOR"}}

                User Query: "{query}"
                Output ONLY JSON with "route": "SQL", "VECTOR", or "HYBRID".
                """
                try:
                    response = groq_client.chat.completions.create(
                        model="openai/gpt-oss-20b",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0,
                        extra_body={"reasoning_effort": "low"},
                    )
                    json_match = re.search(r'\{.*\}', response.choices[0].message.content, re.DOTALL)
                    decision = json.loads(json_match.group(0))
                    route = decision.get("route", "VECTOR")

                    if route in ["SQL", "HYBRID"]:
                        raw_sql = decision.get("sql_query", "")
                        safe_inner_sql = _validate_select_sql(raw_sql)
                        if safe_inner_sql is None:
                            print(f"[ROUTER] Unsafe SQL rejected: {raw_sql!r}")
                            route = "VECTOR"
                        else:
                            allowed_str = "', '".join(allowed_doc_ids)
                            safe_sql = f"SELECT * FROM ({safe_inner_sql}) WHERE doc_id IN ('{allowed_str}')"
                            print(f"[ROUTER] SQL Route: {safe_sql}")
                            df = pd.read_sql_query(safe_sql, self.db_conn)
                            if not df.empty and self._sql_result_relevant(query, df):
                                rendered = self._format_df(df)
                                return {"answer": f"**Data found:**\n\n```\n{rendered}\n```",
                                        "refused": False,
                                        "sources": [{"doc_id": "SQL_DB", "title": "Structured Tables"}],
                                        "confidence": 1.0}
                            elif not df.empty:
                                print(f"[ROUTER] SQL result irrelevant (cols: {list(df.columns)}) — falling back")
                except Exception as e:
                    print(f"[ROUTER ERROR] {e}. Falling back to Vector.")

        if self.db_conn and not is_visual_query:
            id_match = re.search(r"\b([A-Za-z]{1,2}\d{1,2})\b", query)
            if id_match:
                token_upper = id_match.group(1).upper()
                for table_name in self.table_schemas:
                    try:
                        df_full = pd.read_sql_query(f'SELECT * FROM "{table_name}"', self.db_conn)
                    except Exception:
                        continue
                    for col in df_full.columns:
                        if col.lower() in ("doc_id", "chunk_id", "team", "access_level"):
                            continue
                        mask = df_full[col].astype(str).str.upper() == token_upper
                        if mask.any():
                            row_df = df_full[mask]
                            rendered = self._format_df(row_df)
                            print(f"[AUTO-SQL] Direct match for {token_upper} in {table_name}.{col}")
                            return {"answer": f"**Data found:**\n\n```\n{rendered}\n```",
                                    "refused": False,
                                    "sources": [{"doc_id": "SQL_DB", "title": "Structured Tables"}],
                                    "confidence": 1.0}

        print(f"[ROUTER] Vector Route.")
        q_vec = self.embedder.encode(
            [f"Represent this sentence for searching relevant passages: {query}"]
        )[0]
        dense_scores = cosine_similarity([q_vec], self.dense_matrix)[0]

        abbrev_candidates = re.findall(r"\b[A-Z]{2,5}\b", query)
        equation_query = any(
            t in query.lower()
            for t in ("equation", "formula", "derive", "derivation", "expression", "definition")
        )

        image_penalty = 1.0
        if not is_visual_query:
            text_candidates = [i for i in allowed if "image_path" not in self.chunks[i]]
            if text_candidates:
                max_text_sim = max(float(dense_scores[i]) for i in text_candidates)
                if max_text_sim >= 0.35:
                    image_penalty = 0.5

        results = []
        bm25_scores = self.bm25.get_scores(re.findall(r"\w+", query.lower()))
        bm25_rank = {i: r for r, i in enumerate(sorted(allowed, key=lambda i: -bm25_scores[i]))}
        dense_rank = {i: r for r, i in enumerate(sorted(allowed, key=lambda i: -dense_scores[i]))}

        for i in allowed:
            rrf = 1.0 / (60 + bm25_rank[i]) + 1.0 / (60 + dense_rank[i])
            
            # NEW: Table Query Boosting
            if is_table_query:
                if "[Table Data:" in self.chunks[i].get("section_header", ""):
                    rrf *= 10.0  # Overwhelming boost for table data chunks
                elif re.match(r"^table\s*\d+", self.chunks[i].get("text", ""), re.IGNORECASE):
                    rrf *= 0.1   # Demote caption-only chunks
                    
            if image_penalty < 1.0 and "image_path" in self.chunks[i]:
                rrf *= image_penalty
            for cand in abbrev_candidates:
                if re.search(rf"\b{re.escape(cand)}\s*[:=]", self.chunks[i]["text"], re.IGNORECASE):
                    rrf *= 2.0
                    break
            if equation_query and _MATH_LINE.search(self.chunks[i]["text"]):
                rrf *= 1.3
            results.append({"idx": i, "text": self.chunks[i]["text"], "doc": self.chunks[i],
                            "rrf": rrf, "cosine": float(dense_scores[i])})

        results = sorted(results, key=lambda x: -x["rrf"])

        if is_visual_query:
            query_terms = set(re.findall(r"\w+", query.lower())) - _GENERIC_VISUAL
            if query_terms:
                for r in results:
                    caption_terms = set(re.findall(r"\w+", (r["doc"].get("text") or "").lower()))
                    r["_visual_relevance"] = len(caption_terms & query_terms)
                if any(r["_visual_relevance"] > 0 for r in results):
                    for r in results:
                        if r["_visual_relevance"] == 0:
                            print(f"[FILTER] Dropping irrelevant image: {r['doc'].get('image_path')}")
                    results = [r for r in results if r["_visual_relevance"] > 0]

        top_score = float(results[0]["cosine"]) if results else 0.0
        
        is_eq_match = equation_query and results and any(f"({n})" in results[0]["text"] for n in re.findall(r"\b\d+\b", query))
        if top_score < threshold and not is_eq_match:
            return {"answer": "I don't have sufficiently grounded evidence.",
                    "refused": True, "sources": [], "confidence": top_score}

        restricted_hits = [w for w in restricted_keywords if w in query.lower()]
        if restricted_hits:
            chunk_mentions = any(
                any(term in self.chunks[i]["text"].lower() for term in restricted_hits)
                for i in allowed
            )
            if not chunk_mentions:
                return {"answer": f"The accessible engineering documents do not contain "
                                  f"information about {', '.join(restricted_hits)}.",
                        "refused": True, "sources": [], "confidence": top_score}

        unique_results, seen = [], set()
        for r in results:
            if r["doc"]["chunk_id"] in seen:
                continue
            seen.add(r["doc"]["chunk_id"])
            unique_results.append(r)
            if len(unique_results) == 2:
                break

        # NEW: Direct Return for Table Queries
        if is_table_query:
            table_chunks = [r for r in unique_results if "[Table Data:" in r["doc"].get("section_header", "")]
            if table_chunks:
                print("[TABLE BYPASS] Returning table data directly.")
                best_table = table_chunks[0]["doc"]
                return {
                    "answer": self._fix_math_markup(best_table["text"]),
                    "refused": False,
                    "sources": [{"doc_id": best_table["doc_id"], "title": best_table["section_header"]}],
                    "confidence": float(table_chunks[0]["cosine"])
                }

        text_only = [r for r in unique_results if "image_path" not in r["doc"]]
        has_images = any("image_path" in r["doc"] for r in unique_results)

        if text_only and not has_images:
            print(f"[SELECT] Selecting from {len(text_only)} chunk(s)")
            result, used_chunk_ids = self._select_and_assemble(
                query, [r["doc"] for r in text_only]
            )
            if result == self._NO_ANSWER_SENTINEL:
                return {"answer": "Not stated in the report.",
                        "refused": True, "sources": [], "confidence": top_score}
            if result:
                contributing = [
                    r for r in text_only
                    if r["doc"].get("chunk_id") in used_chunk_ids
                    or r["doc"].get("doc_id") in used_chunk_ids
                ]
                return {"answer": result, "refused": False,
                        "sources": [{"doc_id": r["doc"]["doc_id"],
                                     "title": r["doc"].get("section_header", "Unknown")}
                                    for r in contributing],
                        "confidence": top_score}
            print(f"[SELECT] Selector failed — falling back to chunk dump")

        narrative_parts, sources = [], []
        fallback_cache, seen_fallbacks = {}, set()

        for r in unique_results:
            if "image_path" in r["doc"] and Path(r["doc"]["image_path"]).exists():
                print(f"[VQA] Analyzing image: {r['doc']['image_path']}")
                raw_answer = self._answer_with_image(r["doc"]["image_path"], query)
                analysis, is_unknown = self._clean_vqa_answer(raw_answer)

                block = [analysis]
                if is_unknown:
                    block.append("⚠️ *The value is not explicitly labeled on this diagram.*")
                    did = r["doc"]["doc_id"]
                    if did not in seen_fallbacks:
                        seen_fallbacks.add(did)
                        if did not in fallback_cache:
                            fallback_cache[did] = self._lookup_supporting_data(query, did)
                        table_answer = fallback_cache[did]
                        if table_answer:
                            block.append("---")
                            block.append("**Supporting evidence found in the report:**")
                            block.append(table_answer)

                narrative_parts.append(self._fix_math_markup("**Visual Analysis:**\n\n" + "\n\n".join(block)))
                sources.append({
                    "doc_id": r["doc"]["doc_id"],
                    "title": r["doc"].get("text", r["doc"].get("section_header", "Unknown")),
                    "image_path": r["doc"]["image_path"],
                })
            else:
                txt = r["text"].strip()
                if not txt.endswith("."):
                    txt += "."
                narrative_parts.append(self._fix_math_markup(txt))
                sources.append({"doc_id": r["doc"]["doc_id"],
                                "title": r["doc"].get("section_header", "Unknown")})

        if not narrative_parts:
            return {"answer": "The retrieved documents failed verification.",
                    "refused": True, "sources": [], "confidence": 0.0}

        return {"answer": self._fix_math_markup("\n\n".join(narrative_parts)), "refused": False,
                "sources": sources, "confidence": top_score}