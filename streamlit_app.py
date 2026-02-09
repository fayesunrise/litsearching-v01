import hashlib
import json
import re
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd  
import requests
import streamlit as st

##page configuration
st.set_page_config(page_title="Literature searching assistant", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background-color: #fcfcfc;
    }

    .main-title {
        text-align: center;
        color: #1E3A8A; 
        font-weight: 800;
        padding-top: 10px;
        padding-bottom: 5px;
    }
    .sub-title {
        text-align: center;
        color: #64748B;
        margin-bottom: 30px;
        font-size: 1.1rem;
    }

    button[kind="primary"], div.stButton > button[data-testid="stFormSubmitButton"] {
        background-color: #1E3A8A !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    button[kind="primary"]:hover, div.stButton > button[data-testid="stFormSubmitButton"]:hover {
        background-color: #2563EB !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
    }

    div.stButton > button {
        border-radius: 8px;
    }

    div[data-testid="stVerticalBlock"] > div.element-container div.stMarkdown div.stContainer {
        background-color: white;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border: 1px solid #f0f0f0;
        margin-bottom: 20px;
    }

    section[data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<h1 class="main-title">Literature Searching Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Powered by Semantic Scholar and Gemini AI</p>', unsafe_allow_html=True)

def get_secret(name: str) -> str:
    try:
        return st.secrets.get(name, "")
    except Exception:
        return ""

S2_KEY = get_secret("SEMANTIC_SCHOLAR_API_KEY")
GEMINI_KEY = get_secret("GEMINI_API_KEY")

with st.sidebar:
    st.header("Settings")
    st.caption("Configuration from `.streamlit/secrets.toml`")
    if bool(GEMINI_KEY):
        st.success("Gemini API: Connected")
    else:
        st.error("Gemini API: Missing")
    if bool(S2_KEY):
        st.success("Semantic Scholar API: Connected")
    else:
        st.warning("S2 API: Using Rate-Limited Public Access")

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


##S2 API

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


## Gemini API

GEMINI_ENDPOINT_TMPL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"

def _extract_gemini_text(resp_json: Dict[str, Any]) -> str:
    try:
        parts = resp_json["candidates"][0]["content"]["parts"]
        return "\n".join([p["text"] for p in parts if "text" in p]).strip()
    except Exception:
        return json.dumps(resp_json)[:1500]

def gemini_generate_text(contents, api_key, model, system_instruction=None, temperature=0.4, max_output_tokens=1500):
    if not api_key: return {"ok": False, "raw_text": "", "error": "Missing GEMINI_API_KEY"}
    url = GEMINI_ENDPOINT_TMPL.format(model=model, key=api_key)
    # Increased max_output_tokens to 1500 for larger context analysis
    payload = {"contents": contents, "generationConfig": {"temperature": temperature, "maxOutputTokens": max_output_tokens}}
    if system_instruction: payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
    r = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=90)
    if r.status_code != 200: return {"ok": False, "raw_text": r.text[:900], "error": f"Gemini HTTP {r.status_code}"}
    return {"ok": True, "raw_text": _extract_gemini_text(r.json()), "error": None}

def try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    t = (text or "").strip()
    try: return json.loads(t)
    except: pass
    m = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", t, flags=re.DOTALL | re.IGNORECASE)
    if m:
        try: return json.loads(m.group(1).strip())
        except: pass
    return None


## Publication analytic plot

def render_visualizations(papers: List[Dict[str, Any]]):
    if not papers: return
    df = pd.DataFrame(papers)
    
    if 'year' in df.columns:
        st.write("#### Result Insights")
        c1, c2 = st.columns([2, 1])
        with c1:
            st.caption("Publication Trend (Yearly)")
            year_counts = df['year'].dropna().astype(int).value_counts().sort_index()
            st.bar_chart(year_counts, color="#3358BE")
        with c2:
            st.caption("Top Publication Venues")
            venue_counts = df['venue'].replace('', 'Unknown').value_counts().head(5)
            st.dataframe(venue_counts, use_container_width=True)


## Context for synthesis
def build_papers_context(papers, max_papers=None, include_abstract=True, abstract_char_limit=650):
    """
    If max_papers is None, uses all papers in the list.
    """
    chosen = papers if max_papers is None else papers[:max_papers]
    blocks, labels = [], []
    for i, p in enumerate(chosen, start=1):
        label = f"P{i}"
        labels.append(label)
        abstract = clean_ws(p.get("abstract", ""))[:abstract_char_limit] + "..." if include_abstract else ""
        blocks.append(f"[{label}] {p.get('title')}\nYear: {p.get('year')} | Venue: {p.get('venue')}\nAuthors: {format_authors(p.get('authors'))}\nAbstract: {abstract}")
    return "\n---\n".join(blocks).strip(), labels

def make_synthesis_prompt(query: str, context: str) -> str:
    return f"""
User query: {query}

Analyze the provided papers and provide:
1) 5–10 sentence synthesis (cite like [P1]).
2) 8–12 research keywords.
3) 3 refined search queries.
4) Inconsistencies or Controversies: Identify conflicting findings, differing methodologies, or unresolved debates among these papers.

Return ONLY JSON with keys: summary, keywords, refined_queries, controversies.

PAPERS:
{context}
""".strip()

## session initialization
for key in ["saved", "saved_ids", "last_query", "last_results", "chat_messages", "chat_context_signature"]:
    if key not in st.session_state:
        st.session_state[key] = [] if any(x in key for x in ["messages", "saved", "results"]) else set() if "ids" in key else ""


## searching
st.subheader("Search")
with st.form("search_form", clear_on_submit=False):
    col1, col2, col3 = st.columns([6, 1.4, 1.4])
    query = col1.text_input("Keyword query", value=st.session_state.last_query, placeholder="e.g., impact of AI on management")
    limit = col2.selectbox("Results", [5, 10, 20, 30, 50], index=1)
    page = col3.number_input("Page", min_value=1, value=1)
    submitted = st.form_submit_button("Search")

if submitted:
    if not query.strip(): st.warning("Please enter a query.")
    else:
        with st.spinner("Searching..."):
            try:
                results = s2_search_papers(query, S2_KEY, limit=int(limit), offset=(int(page)-1)*int(limit))
                st.session_state.last_query = query
                st.session_state.last_results = results.get("data", [])
            except Exception as e: st.error(f"Error: {e}")


#ai analysis
if st.session_state.last_results:
    with st.expander("View Research Trends", expanded=True):
        render_visualizations(st.session_state.last_results)


st.subheader("AI Insights")
ai_cols = st.columns([3, 3, 4])
target = ai_cols[0].selectbox("Source set", ["Current page results", "Saved papers"])
model_ai = ai_cols[1].text_input("Gemini model", value=DEFAULT_GEMINI_MODEL)


run_ai = st.button("Generate Synthesis!", type="primary", disabled=not bool(GEMINI_KEY))

if run_ai:
    papers = st.session_state.last_results if target == "Current page results" else st.session_state.saved
    if not papers: st.warning("No papers to summarize.")
    else:
        ctx, _ = build_papers_context(papers, max_papers=None)
        prompt = make_synthesis_prompt(st.session_state.last_query, ctx)
        
        with st.spinner(f"Gemini is synthesizing {len(papers)} papers..."):
            res = gemini_generate_text([{"role": "user", "parts": [{"text": prompt}]}], GEMINI_KEY, model_ai, 
                                      system_instruction="Output valid JSON only.")
            if res["ok"]:
                parsed = try_parse_json(res["raw_text"])
                if parsed:
                    st.info("### Synthesis\n" + parsed.get("summary", ""))
                    
                    st.markdown("### Controversies & Inconsistencies")
                    st.warning(parsed.get("controversies", "No major inconsistencies identified in this subset."))
                    
                    st.write("**Keywords:** " + ", ".join(parsed.get("keywords", [])))
                    st.write("**Refined Queries:**")
                    for q in parsed.get("refined_queries", []): st.write(f"- {q}")


## result analysis

st.divider()
if st.session_state.last_results:
    for p in st.session_state.last_results:
        pid = paper_key(p)
        with st.container():
            c1, c2, c3 = st.columns([10, 1.5, 1.5])
            c1.markdown(f"### {p.get('title')}")
            c1.write(f"**{format_authors(p.get('authors'))}** | {p.get('year')} • {p.get('venue')}")
            if c2.button("Save", key=f"s_{md5_key(pid)}", disabled=pid in st.session_state.saved_ids):
                st.session_state.saved.append(p); st.session_state.saved_ids.add(pid); st.toast("Saved!")
            if p.get("openAccessPdf"): c3.link_button("PDF", p["openAccessPdf"]["url"])
            with st.expander("Show Abstract"): st.write(p.get("abstract") or "No abstract available.")


## chat with gemini
st.divider()
st.subheader("Chat with Results?")
if st.session_state.last_results or st.session_state.saved:
    for msg in st.session_state.chat_messages:
        with st.chat_message("assistant" if msg["role"] == "model" else "user"): st.write(msg["text"])
    
    user_input = st.chat_input("Ask a question about these results...")
    if user_input:
        st.session_state.chat_messages.append({"role": "user", "text": user_input})
        with st.chat_message("user"): st.write(user_input)
        papers_chat = st.session_state.last_results if target == "Current page results" else st.session_state.saved
        ctx_chat, _ = build_papers_context(papers_chat, max_papers=None)
        
        full_prompt = f"CONTEXT:\n{ctx_chat}\n\nQUESTION: {user_input}"
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                res = gemini_generate_text([{"role": "user", "parts": [{"text": full_prompt}]}], GEMINI_KEY, model_ai,
                                          system_instruction="Academic assistant. Use [P1] citations. Concise.")
                if res["ok"]:
                    st.write(res["raw_text"])
                    st.session_state.chat_messages.append({"role": "model", "text": res["raw_text"]})