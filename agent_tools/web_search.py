# pip install -U langchain-core langchain-tavily requests ddgs python-dotenv
import os
import threading
import time

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()  # keys (all optional): TAVILY_API_KEY, SERPER_API_KEY

_MAX_RESULTS = 6
_SNIPPET_CHARS = 300
_OUTPUT_CHARS = 4000
_CACHE_TTL = 24 * 3600

_lock = threading.Lock()
_until = {}       # provider -> epoch time it may be used again (inf = off for this run)
_cache = {}       # (query, n) -> (timestamp, text)
_last_call = {}   # provider -> time of its last request slot (pacing)
_tavily_client = None


def _pace(name, gap):
    """Keep at least `gap` seconds between requests to one provider, even across threads."""
    with _lock:
        now = time.monotonic()
        slot = max(now, _last_call.get(name, 0.0) + gap)
        _last_call[name] = slot
    if slot > now:
        time.sleep(slot - now)


def _search_tavily(q, n):
    global _tavily_client
    if _tavily_client is None:
        from langchain_tavily import TavilySearch
        _tavily_client = TavilySearch(max_results=_MAX_RESULTS, topic="general")
    r = _tavily_client.invoke({"query": q})
    if not isinstance(r, dict):
        raise RuntimeError(f"unexpected Tavily response: {str(r)[:100]}")
    if r.get("error"):
        raise RuntimeError(str(r["error"]))
    return [(x.get("title", ""), x.get("url", ""), x.get("content", "")) for x in r.get("results", [])][:n]


def _search_serper(q, n):
    r = requests.post("https://google.serper.dev/search", json={"q": q, "num": n},
                      headers={"X-API-KEY": os.environ["SERPER_API_KEY"]}, timeout=20)
    r.raise_for_status()
    return [(x.get("title", ""), x.get("link", ""), x.get("snippet", "")) for x in r.json().get("organic", [])]


def _search_ddg(q, n):
    from ddgs import DDGS
    return [(x.get("title", ""), x.get("href", ""), x.get("body", "")) for x in DDGS().text(q, max_results=n)]


# (name, env var that must be set or None, function, min seconds between requests)
_PROVIDERS = [
    ("tavily", "TAVILY_API_KEY", _search_tavily, 0.2),
    ("serper", "SERPER_API_KEY", _search_serper, 0.2),
    ("ddg", None, _search_ddg, 1.0),
]


def _mark(name, err):
    """Decide how long to skip a provider after an error. Other errors only skip it for this call."""
    m = str(err).lower()
    if isinstance(err, ImportError):
        wait, why = float("inf"), "package missing (pip install -U langchain-tavily ddgs)"
    elif any(t in m for t in ("401", "402", "unauthorized", "invalid api key", "quota", "credit", "payment")):
        wait, why = float("inf"), "auth/quota"
    elif any(t in m for t in ("429", "rate limit", "ratelimit", "too many requests")):
        wait, why = 60, "rate limit"
    else:
        return
    with _lock:
        _until[name] = time.time() + wait
    print(f"[web_search] {name} skipped ({why}): {str(err)[:120]}")


def _format(items, provider, n):
    seen, lines = set(), []
    for title, url, snippet in items:
        if not url or url in seen:
            continue
        seen.add(url)
        title = " ".join((title or "Untitled").split())
        snippet = " ".join((snippet or "").split())[:_SNIPPET_CHARS]
        lines.append(f"{len(lines) + 1}. {title} | {url} | {snippet}")
        if len(lines) >= n:
            break
    return (f"[via {provider}]\n" + "\n".join(lines))[:_OUTPUT_CHARS] if lines else ""

@tool
def web_fetch(url: str) -> str:
    """Fetch a web page. Returns the text content of the page."""
    if not url:
        return "web_fetch error: empty URL"
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        return r.text[:8000]
    except Exception as e:
        return f"web_fetch error: {str(e)[:100]}"

@tool
def web_search(query: str, max_results: int = 6) -> str:
    """Search the web. Returns numbered 'title | url | snippet' lines.
    Use specific keywords, and open promising URLs with fetch_web_page to verify details."""
    q = " ".join(str(query).split())
    if not q:
        return "web_search error: empty query"
    try:
        n = max(1, min(int(max_results), 10))
    except (TypeError, ValueError):
        n = _MAX_RESULTS
    key = (q.lower(), n)
    with _lock:
        hit = _cache.get(key)
    if hit and time.time() - hit[0] < _CACHE_TTL:
        return hit[1]

    errors = []
    for name, env, fn, gap in _PROVIDERS:
        with _lock:
            off = time.time() < _until.get(name, 0.0)
        if off or (env and not os.getenv(env)):
            continue
        try:
            _pace(name, gap)
            items = fn(q, n)
        except Exception as e:  # never raise: the agent should see the problem as text
            errors.append(f"{name}: {str(e)[:100]}")
            _mark(name, e)
            continue
        text = _format(items, name, n)
        if text:
            with _lock:
                _cache[key] = (time.time(), text)
            return text
        errors.append(f"{name}: no results")
    return "web_search error: " + ("; ".join(errors) or
                                   "no provider available (set TAVILY_API_KEY or SERPER_API_KEY, or install ddgs)")