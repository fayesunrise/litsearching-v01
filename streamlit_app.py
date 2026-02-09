import hashlib
import json
import re
from typing import Dict, Any, List, Optional, Tuple

import requests
import streamlit as st


# =============================
# Page
# =============================
st.set_page_config(page_title="Literature searching assistant", layout="wide")
st.title("Literature searching assistant")
st.write("Meet the paper you want today.")


# =============================
# Secrets (safe handling)
# =============================
def get_secret(name: str) -> str:
    try:
        return st.secrets.get(name, "")
    except Exception:
        return ""


S2_KEY = get_secret("SEMANTIC_SCHOLAR_API_KEY")  # optional (S2 works without key but rate-limited)
GEMINI_KEY = get_secret("GEMINI_API_KEY")        # required for AI functions


with st.sidebar:
    st.header("Settings")
    st.caption("Keys are read from `.streamlit/secrets.toml` (local) or Streamlit Cloud Secrets (deploy).")
    st.write("Semantic Scholar key set:", bool(S2_KEY))
    st.write("Gemini key set:", bool(GEMINI_KEY))
    st.caption("Tip: Do NOT commit `.streamlit/secrets.toml` to GitHub.")


# =============================
# Helpers
# =============================
def md5_key(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]


def format_authors(authors: List[Dict[str, Any]], max_n: int = 6) -> str:
    names = [a.get("name", "") for a in (authors or []) if a.get("name")]
    if len(names) > max_n:
        return ", ".join(names[:max_n]) + ", et al."
    return ", ".join(names) if names else "Unknown authors"


def paper_key(p: Dict[str, Any]) -> str:
    ext = p.get("externalIds") or {}
    return p.get("paperId") or ext.get("DOI") or p.get("title", "paper")


def clean_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


# =============================
# Semantic Scholar API
# =============================
S2_BASE = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS = "paperId,title,abstract,year,venue,authors,citationCount,url,openAccessPdf,externalIds"


@st.cache_data(ttl=3600, show_spinner=False)
def s2_search_papers(query: str, api_key: str, limit: int = 10, offset: int = 0) -> Dict[str, Any]:
    url = f"{S2_BASE}/paper/search"
    params = {"query": query, "limit": limit, "offset": offset, "fields": S2_FIELDS}
    headers = {"x-api-key": api_key} if api_key else {}

    r = requests.get(url, params=params, headers=headers, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"S2 error {r.status_code}: {r.text[:400]}")
    return r.json()


# =============================
# Gemini API (generateContent)
# =============================
GEMINI_ENDPOINT_TMPL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"


def _extract_gemini_text(resp_json: Dict[str, Any]) -> str:
    try:
        parts = resp_json["candidates"][0]["content"]["parts"]
        texts = []
        for part in parts:
            if isinstance(part, dict) and "text" in part:
                texts.append(part["text"])
        return "\n".join(texts).strip()
    except Exception:
        return json.dumps(resp_json)[:1500]


def gemini_generate_text(
    contents: List[Dict[str, Any]],
    api_key: str,
    model: str,
    system_instruction: Optional[str] = None,
    temperature: float = 0.4,
    max_output_tokens: int = 900,
) -> Dict[str, Any]:
    """
    contents: list like [{"role":"user","parts":[{"text":"..."}]}, {"role":"model","parts":[{"text":"..."}]}, ...]
    Returns dict with ok/raw_text/error
    """
    if not api_key:
        return {"ok": False, "raw_text": "", "error": "Missing GEMINI_API_KEY"}

    url = GEMINI_ENDPOINT_TMPL.format(model=model, key=api_key)

    payload: Dict[str, Any] = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        },
    }
    if system_instruction:
        payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

    r = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=60)
    if r.status_code != 200:
        return {"ok": False, "raw_text": r.text[:900], "error": f"Gemini HTTP {r.status_code}"}

    data = r.json()
    text = _extract_gemini_text(data)
    return {"ok": True, "raw_text": text, "error": None}


def _strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", t, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return t


def try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    t = (text or "").strip()
    # direct
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # strip fences
    t2 = _strip_code_fences(t)
    try:
        obj = json.loads(t2)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # extract first {...}
    m = re.search(r"(\{.*\})", t2, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(1))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    return None


# =============================
# Context construction (RAG)
# =============================
def build_papers_context(
    papers: List[Dict[str, Any]],
    max_papers: int = 8,
    include_abstract: bool = True,
    abstract_char_limit: int = 650,
) -> Tuple[str, List[str]]:
    """
    Returns (context_text, paper_labels)
    paper_labels correspond to [P1], [P2], ...
    """
    blocks = []
    labels = []
    chosen = papers[:max_papers]

    for i, p in enumerate(chosen, start=1):
        label = f"P{i}"
        labels.append(label)

        title = clean_ws(p.get("title", "No title"))
        year = p.get("year", "")
        venue = clean_ws(p.get("venue", ""))
        authors = format_authors(p.get("authors", []))
        cites = p.get("citationCount", 0)
        url = p.get("url", "")
        pdf = (p.get("openAccessPdf") or {}).get("url") or ""

        abstract = clean_ws(p.get("abstract", "")) if include_abstract else ""
        if include_abstract and len(abstract) > abstract_char_limit:
            abstract = abstract[:abstract_char_limit] + "..."

        blocks.append(
            f"[{label}] {title}\n"
            f"Year: {year} | Venue: {venue} | Citations: {cites}\n"
            f"Authors: {authors}\n"
            f"URL: {url}\n"
            f"Open PDF: {pdf}\n"
            + (f"Abstract: {abstract}\n" if include_abstract else "")
        )

    return "\n---\n".join(blocks).strip(), labels


def make_synthesis_prompt(query: str, context: str) -> str:
    return f"""
User query: {query}

You are given a list of papers with labels like [P1], [P2], etc.

Task:
1) Write a 5–10 sentence synthesis of the themes across the papers (cite labels like [P3]).
2) Extract 8–12 keywords/keyphrases (specific, research-useful).
3) Suggest 3 refined search queries (more precise than the original).

Return ONLY JSON with keys:
- summary (string)
- keywords (array of strings)
- refined_queries (array of strings)

PAPERS:
{context}
""".strip()


# =============================
# Session state
# =============================
if "saved" not in st.session_state:
    st.session_state.saved = []
if "saved_ids" not in st.session_state:
    st.session_state.saved_ids = set()

if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "last_results" not in st.session_state:
    st.session_state.last_results = []

# Chat history
if "chat_messages" not in st.session_state:
    # Gemini expects roles: "user" and "model"
    st.session_state.chat_messages = []  # list of {"role": "...", "text": "..."}
if "chat_context_signature" not in st.session_state:
    st.session_state.chat_context_signature = ""


# =============================
# SEARCH UI
# =============================
st.subheader("Search")

with st.form("search_form", clear_on_submit=False):
    col1, col2, col3 = st.columns([6, 1.4, 1.4])
    query = col1.text_input(
        "Keyword query",
        value=st.session_state.last_query,
        placeholder="e.g., red teaming language models",
    )
    limit = col2.selectbox("Results", [5, 10, 20], index=1)
    page = col3.number_input("Page", min_value=1, value=1, step=1)
    show_raw = st.checkbox("Show raw Semantic Scholar response", value=False)
    submitted = st.form_submit_button("Search", type="primary")

if submitted:
    q = (query or "").strip()
    if not q:
        st.warning("Please enter a search query.")
    else:
        offset = (int(page) - 1) * int(limit)
        with st.spinner("Searching Semantic Scholar..."):
            try:
                results = s2_search_papers(q, S2_KEY, limit=int(limit), offset=int(offset))
            except Exception as e:
                st.error(f"Search failed: {e}")
                st.stop()

        st.session_state.last_query = q
        st.session_state.last_results = results.get("data", [])
        total = results.get("total", 0)
        st.caption(f"Found ~{total} results. Showing {len(st.session_state.last_results)} results (page {page}).")

        if show_raw:
            st.json(results)


# =============================
# AI SYNTHESIS (JSON)
# =============================
st.subheader("AI synthesis (Gemini)")

ai_cols = st.columns([2.2, 2.2, 2.2, 3.4])
target = ai_cols[0].selectbox("Summarize which set?", ["Current page results", "Saved papers"], index=0)
model = ai_cols[1].text_input("Gemini model", value=DEFAULT_GEMINI_MODEL)
max_p = ai_cols[2].selectbox("Use top-N papers", [5, 8, 10], index=1)
show_ai_raw = ai_cols[3].checkbox("Show AI raw output", value=False)

run_ai = st.button("AI: Summarize + Keywords", type="primary", disabled=not bool(GEMINI_KEY))

if run_ai:
    papers = st.session_state.last_results if target == "Current page results" else st.session_state.saved
    if not papers:
        st.warning("No papers to summarize yet. Run a search (or save papers) first.")
    else:
        ctx, _labels = build_papers_context(papers, max_papers=int(max_p), include_abstract=True)
        prompt = make_synthesis_prompt(st.session_state.last_query, ctx)

        with st.spinner("Calling Gemini..."):
            res = gemini_generate_text(
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
                api_key=GEMINI_KEY,
                model=model,
                system_instruction="You output ONLY valid JSON. No code fences. No extra text.",
                temperature=0.4,
                max_output_tokens=900,
            )

        if not res["ok"]:
            st.error(res["error"])
            st.code(res.get("raw_text", ""), language="text")
        else:
            raw = res["raw_text"]
            parsed = try_parse_json(raw)

            if show_ai_raw:
                st.markdown("### AI raw output")
                st.code(raw, language="text")

            if not parsed:
                st.error("Gemini did not return valid JSON. Turn on 'Show AI raw output' to inspect.")
            else:
                st.success("AI synthesis generated.")
                st.markdown("### Summary")
                st.write(parsed.get("summary", ""))

                st.markdown("### Keywords")
                kws = parsed.get("keywords", [])
                if isinstance(kws, list) and kws:
                    st.write(", ".join([str(x) for x in kws]))
                else:
                    st.write(kws)

                st.markdown("### Refined queries")
                rq = parsed.get("refined_queries", [])
                if isinstance(rq, list):
                    for i, x in enumerate(rq, start=1):
                        st.write(f"{i}. {x}")
                else:
                    st.write(rq)

if not GEMINI_KEY:
    st.info("To enable AI features, add `GEMINI_API_KEY` in `.streamlit/secrets.toml` or Streamlit Cloud Secrets.")


# =============================
# RESULTS LIST
# =============================
st.divider()
st.subheader("Results")

if not st.session_state.last_results:
    st.info("Run a search above to see results here.")
else:
    for p in st.session_state.last_results:
        pid = paper_key(p)
        kid = md5_key("res_" + pid)

        title = p.get("title", "No title")
        year = p.get("year", "")
        venue = p.get("venue", "")
        authors = format_authors(p.get("authors", []))
        cites = p.get("citationCount", 0)
        url = p.get("url", "")
        abstract = p.get("abstract") or ""
        pdf = (p.get("openAccessPdf") or {}).get("url")

        with st.container(border=True):
            top = st.columns([10, 1.6, 1.6])

            top[0].markdown(f"### {title}")
            top[0].write(f"**{authors}**")
            top[0].write(f"{year} • {venue} • Citations: {cites}")

            if url:
                top[0].link_button("Open in Semantic Scholar", url)

            save_disabled = pid in st.session_state.saved_ids
            if top[1].button("Save", key=f"save_{kid}", disabled=save_disabled):
                st.session_state.saved.append(p)
                st.session_state.saved_ids.add(pid)
                st.toast("Saved!", icon="✅")

            if pdf:
                top[2].link_button("Open PDF", pdf)
            else:
                top[2].button("No PDF", key=f"nopdf_{kid}", disabled=True)

            with st.expander("Abstract"):
                st.write(abstract if abstract else "No abstract available.")


# =============================
# SAVED LIST
# =============================
st.divider()
st.subheader("Saved papers")

if not st.session_state.saved:
    st.info("No saved papers yet. Search above and click **Save** on papers you like.")
else:
    for i, p in enumerate(st.session_state.saved, start=1):
        pid = paper_key(p)
        kid = md5_key("saved_" + pid)

        title = p.get("title", "No title")
        year = p.get("year", "")
        authors = format_authors(p.get("authors", []))
        url = p.get("url", "")

        with st.container(border=True):
            cols = st.columns([10, 1.5])
            cols[0].markdown(f"**{i}. {title}**")
            cols[0].write(f"{authors} • {year}")
            if url:
                cols[0].link_button("Open", url)

            if cols[1].button("Remove", key=f"rm_{kid}"):
                st.session_state.saved = [x for x in st.session_state.saved if paper_key(x) != pid]
                st.session_state.saved_ids.discard(pid)
                st.rerun()

    if st.button("Clear all saved"):
        st.session_state.saved = []
        st.session_state.saved_ids = set()
        st.rerun()


# =============================
# CHAT WITH RESULTS (Scholar-lab style)
# =============================
st.divider()
st.subheader("Chat with your search results (Gemini)")

chat_cols = st.columns([2.4, 2.4, 2.0, 3.2])
chat_target = chat_cols[0].selectbox("Chat over", ["Current page results", "Saved papers"], index=0)
chat_model = chat_cols[1].text_input("Gemini model (chat)", value=DEFAULT_GEMINI_MODEL)
chat_topn = chat_cols[2].selectbox("Context papers (top N)", [5, 8, 10], index=1)
chat_include_abs = chat_cols[3].checkbox("Include abstracts in context", value=True)

# Build context + signature (so if context changes, we can warn/clear)
papers_for_chat = st.session_state.last_results if chat_target == "Current page results" else st.session_state.saved
context_text, labels = build_papers_context(
    papers_for_chat,
    max_papers=int(chat_topn),
    include_abstract=bool(chat_include_abs),
)

context_signature = md5_key(chat_target + str(chat_topn) + str(chat_include_abs) + str(len(papers_for_chat)) + (st.session_state.last_query or ""))
if st.session_state.chat_context_signature and st.session_state.chat_context_signature != context_signature:
    st.warning("Your chat context changed (different papers/settings). Consider clearing chat for consistency.")

btn_cols = st.columns([1.2, 1.2, 7.6])
clear_chat = btn_cols[0].button("Clear chat", disabled=False)
show_context = btn_cols[1].button("Show context", disabled=(not bool(context_text)))
if show_context:
    st.code(context_text or "(no context yet)", language="text")

if clear_chat:
    st.session_state.chat_messages = []
    st.session_state.chat_context_signature = context_signature
    st.rerun()

# If no papers, chat won't be useful
if not papers_for_chat:
    st.info("Chat needs papers. Run a search first (or save papers), then come back here.")
else:
    # render history
    for m in st.session_state.chat_messages:
        role = "assistant" if m["role"] == "model" else "user"
        with st.chat_message(role):
            st.write(m["text"])

    user_q = st.chat_input("Ask a question about these results…", disabled=not bool(GEMINI_KEY))

    if user_q:
        # Update signature at first message
        if not st.session_state.chat_context_signature:
            st.session_state.chat_context_signature = context_signature

        # add user message
        st.session_state.chat_messages.append({"role": "user", "text": user_q})
        with st.chat_message("user"):
            st.write(user_q)

        # Build Gemini contents from recent history (keep it short)
        # Gemini supports multi-turn with roles user/model
        recent = st.session_state.chat_messages[-10:]
        contents = []
        for msg in recent:
            contents.append({"role": msg["role"], "parts": [{"text": msg["text"]}]})

        # System instruction: force grounding + citations
        system_instruction = (
            "You are an academic literature assistant. "
            "You MUST ground answers only in the provided paper context. "
            "When you make a claim, cite the relevant papers using labels like [P1], [P2]. "
            "If the answer is not supported by the context, say you don't know and suggest a follow-up search query. "
            "Keep answers concise and actionable."
        )

        # We inject context as a prefix each time (simple & reliable)
        context_prefix = (
            f"CONTEXT PAPERS:\n{context_text}\n\n"
            "Remember: cite with [P#] labels.\n"
        )

        # Replace the last user message text with context+question (so model sees context)
        # (Do not mutate the stored chat history; just modify payload)
        contents_for_call = contents[:-1] + [
            {"role": "user", "parts": [{"text": context_prefix + "\nUSER QUESTION:\n" + user_q}]}
        ]

        with st.chat_message("assistant"):
            with st.spinner("Gemini is thinking…"):
                res = gemini_generate_text(
                    contents=contents_for_call,
                    api_key=GEMINI_KEY,
                    model=chat_model,
                    system_instruction=system_instruction,
                    temperature=0.3,
                    max_output_tokens=900,
                )

            if not res["ok"]:
                st.error(res["error"])
                st.code(res.get("raw_text", ""), language="text")
            else:
                answer = res["raw_text"].strip()
                st.write(answer)
                st.session_state.chat_messages.append({"role": "model", "text": answer})

    if not GEMINI_KEY:
        st.info("Add `GEMINI_API_KEY` to enable chat.")
