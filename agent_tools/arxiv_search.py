# pip install -U arxiv langchain-core
import re
import threading
import time

import arxiv
from langchain_core.tools import tool

_CACHE_TTL = 24 * 3600
_OUTPUT_CHARS = 4000
_ABSTRACT_CHARS = 500
_MIN_GAP = 3.1        # arXiv asks for at most one API request every ~3 seconds
_BLOCK_SECONDS = 30   # pause after the API reports overload (HTTP 429/503)

_client = arxiv.Client(page_size=10, delay_seconds=3, num_retries=2)
_api_lock = threading.Lock()   # one arXiv request at a time, across all threads
_lock = threading.Lock()
_cache = {}                    # (query, n, sort) -> (timestamp, text)
_state = {"last": 0.0, "blocked_until": 0.0}

_ID = r"(?:\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?"
_ID_RE = re.compile(rf"^(?:arxiv:|https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/)?({_ID})(?:\.pdf)?$", re.I)


def _parse_ids(q):
    """'2405.06707', 'arXiv:2405.06707v2', abs/pdf URLs, old-style 'hep-th/9901001', or several separated by commas/spaces."""
    parts = [p for p in re.split(r"[,\s]+", q.strip()) if p]
    ids = [m.group(1) for p in parts if (m := _ID_RE.match(p))]
    return ids if parts and len(ids) == len(parts) else None


def _fix_dates(q):
    """Turn 2022-01-31 or 20220131 inside submittedDate/lastUpdatedDate ranges into arXiv's YYYYMMDDHHMM."""
    def one(s, end):
        s = s.strip()
        m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", s)
        if m:
            s = "".join(m.groups())
        return s + ("2359" if end else "0000") if re.fullmatch(r"\d{8}", s) else s

    def repl(m):
        return f"{m.group(1)}:[{one(m.group(2), False)} TO {one(m.group(3), True)}]"

    return re.sub(r"(submittedDate|lastUpdatedDate):\[\s*([^\]]+?)\s+TO\s+([^\]]+?)\s*\]", repl, q, flags=re.I)


def _format(results):
    out = []
    for i, r in enumerate(results, 1):
        authors = ", ".join(a.name for a in r.authors[:6]) + (" et al." if len(r.authors) > 6 else "")
        sid = r.get_short_id()
        line = (f"{i}. {' '.join(r.title.split())} | {authors} | submitted {r.published.date()}"
                f" (updated {r.updated.date()}) | {r.primary_category}\n"
                f"   id {sid} | abs https://arxiv.org/abs/{sid} | PDF {r.pdf_url}")
        if getattr(r, "journal_ref", None):
            line += f"\n   journal: {' '.join(r.journal_ref.split())[:100]}"
        line += f"\n   {' '.join(r.summary.split())[:_ABSTRACT_CHARS]}"
        out.append(line)
    return "\n".join(out)[:_OUTPUT_CHARS]


@tool
def arxiv_search(query: str, max_results: int = 5, sort: str = "relevance") -> str:
    """Search arXiv papers. query can be plain keywords, field searches (au:smith, ti:"graph networks",
    abs:agents, cat:cs.AI) joined with AND/OR, a date window like
    submittedDate:[2022-01-01 TO 2022-01-31], or one or more arXiv ids/URLs (2405.06707, arxiv.org/abs/...).
    sort: 'relevance', 'newest' or 'oldest' (by first submission date).
    Returns title, authors, first-submission date, last update, category, id, abs/PDF links, journal
    reference if any, and the abstract. For the full text or version history, open the abs or PDF link
    with fetch_web_page."""
    q = " ".join(str(query).split())
    if not q:
        return "arxiv_search error: empty query"
    try:
        n = max(1, min(int(max_results), 10))
    except (TypeError, ValueError):
        n = 5
    s = str(sort).lower()
    mode = "newest" if s.startswith(("new", "date")) else "oldest" if s.startswith("old") else "relevance"
    key = (q.lower(), n, mode)
    with _lock:
        hit = _cache.get(key)
        blocked = _state["blocked_until"] - time.time()
    if hit and time.time() - hit[0] < _CACHE_TTL:
        return hit[1]
    if blocked > 0:
        return (f"arxiv_search is paused for ~{int(blocked) + 1}s (arXiv API overloaded). Meanwhile use "
                "fetch_web_page on https://arxiv.org/abs/<id> or web_search.")

    try:
        ids = _parse_ids(q)
        if ids:
            search = arxiv.Search(id_list=ids, max_results=len(ids))
        else:
            order = {"newest": (arxiv.SortCriterion.SubmittedDate, arxiv.SortOrder.Descending),
                     "oldest": (arxiv.SortCriterion.SubmittedDate, arxiv.SortOrder.Ascending),
                     "relevance": (arxiv.SortCriterion.Relevance, arxiv.SortOrder.Descending)}[mode]
            search = arxiv.Search(query=_fix_dates(q), max_results=n, sort_by=order[0], sort_order=order[1])
        with _api_lock:
            wait = _state["last"] + _MIN_GAP - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            try:
                results = list(_client.results(search))
            finally:
                _state["last"] = time.monotonic()
    except Exception as e:  # never raise: the agent should see the problem as text
        msg = str(e)
        if getattr(e, "status", None) in (429, 503) or any(t in msg for t in ("429", "503")):
            with _lock:
                _state["blocked_until"] = time.time() + _BLOCK_SECONDS
            print(f"[arxiv_search] API overloaded, pausing {_BLOCK_SECONDS}s: {msg[:120]}")
        return f"arxiv_search error: {msg[:200]}"

    if not results:
        return "No arXiv results. Try fewer or different keywords, or search by author with au:Lastname."
    text = _format(results)
    with _lock:
        _cache[key] = (time.time(), text)
    return text