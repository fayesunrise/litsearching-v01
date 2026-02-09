import streamlit as st
import requests
from typing import Dict, Any, List
import os, pathlib, time, json

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Literature searching assistant", layout="wide")
st.title("Literature searching assistant")
st.write("Meet the paper you want today.")

# Optional debug (remove later)
with st.expander("Debug (optional)", expanded=False):
    st.info(f"Running file: {__file__}")
    st.info(f"CWD: {os.getcwd()}")
    st.info(f"File mtime: {time.ctime(pathlib.Path(__file__).stat().st_mtime)}")

# -----------------------------
# Secrets / keys (safe handling)
# -----------------------------
def get_secret(name: str) -> str:
    """Return secret value if present, else empty string (doesn't crash if secrets missing)."""
    try:
        return st.secrets.get(name, "")
    except Exception:
        return ""

S2_KEY = get_secret("SEMANTIC_SCHOLAR_API_KEY")

with st.sidebar:
    st.header("Settings")
    st.caption("Keys are read from `.streamlit/secrets.toml` (local) or Streamlit Cloud Secrets (deploy).")
    st.write("Semantic Scholar key set:", bool(S2_KEY))
    if not S2_KEY:
        st.warning("No Semantic Scholar API key found. Add it to `.streamlit/secrets.toml`.")

# -----------------------------
# Semantic Scholar API helpers
# -----------------------------
S2_BASE = "https://api.semanticscholar.org/graph/v1"
FIELDS = "paperId,title,abstract,year,venue,authors,citationCount,url,openAccessPdf,externalIds"

@st.cache_data(ttl=3600, show_spinner=False)
def s2_search_papers(query: str, api_key: str, limit: int = 10, offset: int = 0) -> Dict[str, Any]:
    """Search papers by keyword query."""
    url = f"{S2_BASE}/paper/search"
    params = {"query": query, "limit": limit, "offset": offset, "fields": FIELDS}
    headers = {"x-api-key": api_key} if api_key else {}

    r = requests.get(url, params=params, headers=headers, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"S2 error {r.status_code}: {r.text[:500]}")
    return r.json()

def format_authors(authors: List[Dict[str, Any]], max_n: int = 6) -> str:
    names = [a.get("name", "") for a in (authors or []) if a.get("name")]
    if len(names) > max_n:
        return ", ".join(names[:max_n]) + ", et al."
    return ", ".join(names) if names else "Unknown authors"

def paper_key(p: Dict[str, Any]) -> str:
    """Stable id for saving/removing."""
    return (
        p.get("paperId")
        or (p.get("externalIds") or {}).get("DOI")
        or (p.get("externalIds") or {}).get("ArXiv")
        or p.get("title", "paper")
    )

# -----------------------------
# App state
# -----------------------------
if "saved" not in st.session_state:
    st.session_state.saved = []
if "saved_ids" not in st.session_state:
    st.session_state.saved_ids = set()

# -----------------------------
# Search UI (IMPORTANT FIX: use a form, but DO NOT disable the submit button)
# -----------------------------
st.subheader("Search")

with st.form("search_form"):
    col1, col2, col3 = st.columns([6, 1.2, 1.2])
    query = col1.text_input(
        "Keyword query",
        placeholder="e.g., red teaming, human-AI teaming predictability, LLM political bias",
        key="query",
    )
    limit = col2.selectbox("Results", [5, 10, 20], index=1, key="limit")
    page = col3.number_input("Page", min_value=1, value=1, step=1, key="page")
    show_raw = st.checkbox("Show raw response", key="show_raw")

    submitted = st.form_submit_button("Search", type="primary")  # ✅ always enabled

# -----------------------------
# Results
# -----------------------------
if submitted:
    q = st.session_state.get("query", "").strip()
    lim = int(st.session_state.get("limit", 10))
    pg = int(st.session_state.get("page", 1))
    offset = (pg - 1) * lim

    if not q:
        st.warning("Please enter a search query.")
        st.stop()

    if not S2_KEY:
        st.error("Missing SEMANTIC_SCHOLAR_API_KEY. Add it to `.streamlit/secrets.toml` then rerun.")
        st.stop()

    with st.spinner("Searching Semantic Scholar..."):
        try:
            results = s2_search_papers(q, S2_KEY, limit=lim, offset=offset)
        except Exception as e:
            st.error(f"Search failed: {e}")
            st.stop()

    total = results.get("total", 0)
    data = results.get("data", [])

    st.caption(f"Found ~{total} results. Showing {len(data)} results (page {pg}).")

    if st.session_state.get("show_raw"):
        st.code(json.dumps(results, indent=2)[:8000], language="json")

    for p in data:
        pid = paper_key(p)
        title = p.get("title", "No title")
        year = p.get("year", "")
        venue = p.get("venue", "")
        authors = format_authors(p.get("authors", []))
        cites = p.get("citationCount", 0)
        url = p.get("url", "")
        abstract = p.get("abstract", "")

        with st.container(border=True):
            top = st.columns([10, 1.6, 1.6])
            top[0].markdown(f"### {title}")
            top[0].write(f"**{authors}**")
            top[0].write(f"{year} • {venue} • Citations: {cites}")
            if url:
                top[0].write(url)

            save_disabled = pid in st.session_state.saved_ids
            if top[1].button("Save", key=f"save_{pid}", disabled=save_disabled):
                st.session_state.saved.append(p)
                st.session_state.saved_ids.add(pid)
                st.toast("Saved!", icon="✅")

            pdf = (p.get("openAccessPdf") or {}).get("url")
            if pdf:
                top[2].link_button("Open PDF", pdf)
            else:
                top[2].button("No PDF", key=f"nopdf_{pid}", disabled=True)

            with st.expander("Abstract"):
                st.write(abstract if abstract else "No abstract available.")

# -----------------------------
# Saved list
# -----------------------------
st.divider()
st.subheader("Saved papers")

if not st.session_state.saved:
    st.info("No saved papers yet. Search above and click **Save** on papers you like.")
else:
    for i, p in enumerate(st.session_state.saved, start=1):
        pid = paper_key(p)
        title = p.get("title", "No title")
        year = p.get("year", "")
        authors = format_authors(p.get("authors", []))
        url = p.get("url", "")

        with st.container(border=True):
            cols = st.columns([10, 1.5])
            cols[0].markdown(f"**{i}. {title}**")
            cols[0].write(f"{authors} • {year}")
            if url:
                cols[0].write(url)

            if cols[1].button("Remove", key=f"rm_{pid}"):
                st.session_state.saved = [x for x in st.session_state.saved if paper_key(x) != pid]
                st.session_state.saved_ids.discard(pid)
                st.rerun()

    if st.button("Clear all saved"):
        st.session_state.saved = []
        st.session_state.saved_ids = set()
        st.rerun()
