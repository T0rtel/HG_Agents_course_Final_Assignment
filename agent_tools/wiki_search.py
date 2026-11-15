# pip install -U langchain-core requests beautifulsoup4
import calendar
import datetime
import html as htmlmod
import os
import re
import threading
import time
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool

_API = "https://en.wikipedia.org/w/api.php"
# Wikimedia asks API clients to identify themselves: set WIKI_CONTACT to your email or a project URL.
_HEADERS = {"User-Agent": f"gaia-agent/1.0 (course project; contact: {os.getenv('WIKI_CONTACT', 'unset')})"}
_CHUNK = 6000            # characters per call (~1.5k tokens); raise to 15000+ if you're on Gemini
_CACHE_TTL = 24 * 3600
_MIN_GAP = 0.1           # seconds between API requests, across threads
_BLOCK_SECONDS = 20      # pause after HTTP 429/503
_MAX_REV_PAGES = 60      # history cap: 60 pages x 500 revisions
_SKIP = {"references", "external links", "further reading", "see also", "notes", "citations"}
_NS = {"": "0", "article": "0", "talk": "1", "user": "2", "wikipedia": "4", "file": "6", "template": "10",
       "help": "12", "category": "14", "portal": "100", "draft": "118", "all": "*"}

_lock = threading.Lock()
_pages = {}      # (title, revid) -> (timestamp, (title, revid, sections))
_searches = {}   # (query, limit, namespace) -> (timestamp, text)
_state = {"last": 0.0, "blocked_until": 0.0}


class _Overloaded(Exception):
    pass


def _api(**params):
    with _lock:
        left = _state["blocked_until"] - time.time()
        if left > 0:
            raise _Overloaded(f"Wikipedia API paused for ~{int(left) + 1}s after an overload response")
        now = time.monotonic()
        slot = max(now, _state["last"] + _MIN_GAP)
        _state["last"] = slot
    if slot > now:
        time.sleep(slot - now)
    params.update(format="json", formatversion=2)
    r = requests.get(_API, params=params, headers=_HEADERS, timeout=30)
    if r.status_code in (429, 503):
        ra = str(r.headers.get("Retry-After", ""))
        wait = min(int(ra), 120) if ra.isdigit() else _BLOCK_SECONDS
        with _lock:
            _state["blocked_until"] = time.time() + wait
        raise _Overloaded(f"Wikipedia API busy (HTTP {r.status_code}); paused {wait}s")
    r.raise_for_status()
    d = r.json()
    if "error" in d:
        raise RuntimeError(f"Wikipedia API error: {d['error'].get('info', d['error'])}")
    return d


def _guard(name, fn):
    """Tools never raise: the agent should see problems as text."""
    try:
        return fn()
    except _Overloaded as e:
        return f"{name} error: {e}. Meanwhile use fetch_web_page on the en.wikipedia.org URL."
    except Exception as e:
        return f"{name} error: {str(e)[:200]}"


def _bound(s, end):
    """'2022' / '2022-06' / '2022-06-15' -> ISO timestamp (UTC) at the start or end of that period."""
    m = re.fullmatch(r"(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?", s.strip())
    if not m:
        raise ValueError("dates must look like 2022, 2022-06 or 2022-06-15")
    y = int(m.group(1))
    if not m.group(2):
        mo, d = (12, 31) if end else (1, 1)
    else:
        mo = int(m.group(2))
        d = int(m.group(3)) if m.group(3) else (calendar.monthrange(y, mo)[1] if end else 1)
    datetime.date(y, mo, d)  # validates the date
    return f"{y:04d}-{mo:02d}-{d:02d}T{'23:59:59' if end else '00:00:00'}Z"


def _txt(el):
    t = el.get_text(" ", strip=True)
    t = re.sub(r"\s+([,.;:!?)\]])", r"\1", t)
    t = re.sub(r"([(\[])\s+", r"\1", t)
    return re.sub(r"\s{2,}", " ", t)


def _table(tb):
    if "infobox" in (tb.get("class") or []):
        out = []
        for tr in tb.find_all("tr"):
            th, td = tr.find("th"), tr.find("td")
            if th and td:
                out.append(f"{_txt(th)}: {_txt(td)}")
            elif th or td:
                out.append(_txt(th or td))
        return "[Infobox]\n" + "\n".join(o for o in out if o.strip(": "))
    grid, pending = [], {}  # pending carries rowspan cells down to later rows
    trs = [tr for tr in tb.find_all("tr") if tr.find_parent("table") is tb]
    for r, tr in enumerate(trs):
        row, c = [], 0
        for cell in tr.find_all(["th", "td"], recursive=False):
            while (r, c) in pending:
                row.append(pending.pop((r, c)))
                c += 1
            text = _txt(cell)
            rs = int(re.sub(r"\D", "", cell.get("rowspan", "1")) or 1)
            cs = int(re.sub(r"\D", "", cell.get("colspan", "1")) or 1)
            for _ in range(cs):
                row.append(text)
                for dr in range(1, rs):
                    pending[(r + dr, c)] = text
                c += 1
        while (r, c) in pending:
            row.append(pending.pop((r, c)))
            c += 1
        if any(row):
            grid.append(row)
    cap = tb.find("caption")
    head = f"[Table: {_txt(cap)}]" if cap else "[Table]"
    return "\n".join([head] + [" | ".join(row) for row in grid])


def _sections(html):
    soup = BeautifulSoup(html, "html.parser")
    for h in soup.select(".hatnote"):  # keep "Main article: ..." pointers as plain paragraphs
        h.name = "p"
    for fig in soup.select("figure, div.thumb"):  # keep image captions, drop the rest of the image markup
        cap = fig.select_one("figcaption, .thumbcaption")
        text = _txt(cap) if cap else ""
        if text:
            p = soup.new_tag("p")
            p.string = "[Image] " + text
            fig.replace_with(p)
        else:
            fig.decompose()
    for sel in ("sup.reference", ".mw-editsection", "style", "script", ".navbox", ".vertical-navbox",
                ".sistersitebox", ".metadata", ".mw-empty-elt", "#toc", ".toc"):
        for el in soup.select(sel):
            el.decompose()
    for br in soup.find_all("br"):
        br.replace_with(" / ")
    root = soup.select_one("div.mw-parser-output") or soup
    secs = [{"title": "Lead", "level": 1, "lines": []}]
    for el in root.find_all(["h2", "h3", "h4", "h5", "h6", "p", "ul", "ol", "dl", "table", "blockquote"]):
        if el.name[0] == "h":
            secs.append({"title": _txt(el), "level": int(el.name[1]), "lines": []})
            continue
        if el.find_parent(["table", "li", "dd", "blockquote"]):
            continue  # already covered by its parent element
        if el.name == "table":
            text = _table(el)
        elif el.name in ("ul", "ol"):
            text = "\n".join("- " + _txt(li) for li in el.find_all("li", recursive=False))
        else:
            text = _txt(el)
        if text.strip():
            secs[-1]["lines"].append(text)
    return secs


def _resolve(query):
    q = " ".join(query.split())
    d = _api(action="query", titles=q, redirects=1, prop="pageprops", ppprop="disambiguation")
    page = d["query"]["pages"][0]
    if not page.get("missing") and not page.get("invalid"):
        return page["title"], "disambiguation" in page.get("pageprops", {}), []
    hits = _api(action="query", list="search", srsearch=q, srlimit=5)["query"]["search"]
    if not hits:
        return None, False, []
    titles = [h["title"] for h in hits]
    return titles[0], False, titles[1:]


def _revision(title, as_of):
    d = _api(action="query", prop="revisions", titles=title, rvlimit=1, rvdir="older",
             rvstart=_bound(as_of, True), rvprop="ids|timestamp")
    revs = d["query"]["pages"][0].get("revisions")
    if not revs:
        raise ValueError(f"'{title}' has no revision on or before {as_of} (it may have been created or renamed later)")
    return revs[0]["revid"], revs[0]["timestamp"]


def _load(title, revid):
    key = (title, revid)
    with _lock:
        hit = _pages.get(key)
    if hit and time.time() - hit[0] < _CACHE_TTL:
        return hit[1]
    kw = {"oldid": revid} if revid else {"page": title, "redirects": 1}
    p = _api(action="parse", prop="text", disableeditsection=1, disablelimitreport=1, **kw)["parse"]
    val = (p["title"], p["revid"], _sections(p["text"]))
    with _lock:
        _pages[key] = (time.time(), val)
    return val


def _render(secs, section):
    def fmt(block):
        out = []
        for s in block:
            if s["title"] != "Lead":
                out.append("\n" + "#" * s["level"] + " " + s["title"])
            out.extend(s["lines"])
        return "\n".join(out).strip()

    if not section:
        return fmt([s for s in secs if s["title"].lower() not in _SKIP])
    q = section.lower().strip()
    idx = next((i for i, s in enumerate(secs) if s["title"].lower() == q), None)
    if idx is None:
        idx = next((i for i, s in enumerate(secs) if q in s["title"].lower()), None)
    if idx is None:
        return None
    block = [secs[idx]]
    for t in secs[idx + 1:]:
        if t["level"] <= secs[idx]["level"]:
            break
        block.append(t)
    return fmt(block)


def _page(text, offset):
    if not text.strip():
        return "\n(no readable text)"
    if offset >= len(text):
        return f"\n[offset {offset} is past the end ({len(text)} chars)]"
    end = min(offset + _CHUNK, len(text))
    chunk = text[offset:end]
    if end < len(text) and "\n" in chunk[_CHUNK // 2:]:
        chunk = chunk[: chunk.rfind("\n")]  # cut at a line/row boundary
        end = offset + len(chunk)
    foot = f"\n[chars {offset}-{end} of {len(text)}; " + (
        f"call again with offset={end} to continue]" if end < len(text) else "end]")
    return f"\n\n{chunk}{foot}"


def _lookup(title, as_of, section, offset):
    page, disambig, others = _resolve(title)
    if not page:
        return f"No Wikipedia page found for '{title}'. Try wikipedia_search with different words."
    revid = ts = None
    if as_of:
        revid, ts = _revision(page, as_of)
    name, revid, secs = _load(page, revid)
    url = f"https://en.wikipedia.org/w/index.php?title={quote(name.replace(' ', '_'))}&oldid={revid}"
    head = f"# {name}\nSource: {url} " + (f"(version as of {ts})" if ts else "(current version)")
    if name.lower() != " ".join(title.split()).lower():
        head += f"\nNote: '{title}' resolved to '{name}'."
    if others:
        head += " Other search hits: " + ", ".join(others) + "."
    if disambig:
        head += "\nThis is a disambiguation page; call again with one of the specific titles below."
    text = _render(secs, section)
    if text is None:
        return f"{head}\nSection '{section}' not found. Sections: " + ", ".join(s["title"] for s in secs)
    if offset == 0 and not section:
        toc = "; ".join(f"{s['title']} ({sum(len(x) for x in s['lines'])} chars)"
                        for s in secs if s["title"].lower() not in _SKIP)
        head += f"\nContents: {toc}"
    return head + _page(text, offset)


def _history(title, start, end, list_limit):
    page, _, _ = _resolve(title)
    if not page:
        return f"No Wikipedia page found for '{title}'. Try wikipedia_search with different words."
    params = dict(action="query", prop="revisions", titles=page, rvlimit="max", rvdir="newer",
                  rvprop="ids|timestamp|user" + ("|comment" if list_limit else ""))
    if start:
        params["rvstart"] = _bound(start, False)
    if end:
        params["rvend"] = _bound(end, True)
    total, users, first, last, listing, cont, pages, capped = 0, set(), None, None, [], {}, 0, False
    while True:
        d = _api(**params, **cont)
        for r in d["query"]["pages"][0].get("revisions", []):
            total += 1
            who = r.get("user", "(hidden)")
            users.add(who)
            first = first or (r["timestamp"], who)
            last = (r["timestamp"], who)
            if len(listing) < list_limit:
                listing.append(f"{len(listing) + 1}. {r['timestamp']} | {who} | {' '.join(r.get('comment', '').split())[:80]}")
        pages += 1
        cont = d.get("continue")
        if not cont:
            break
        if pages >= _MAX_REV_PAGES:
            capped = True
            break
    rng = f"{start or 'page creation'} to {end or 'now'}"
    head = f"# Edit history: {page}\nRange (UTC, inclusive, end date runs to 23:59:59): {rng}"
    if not total:
        return head + "\nRevisions in range: 0"
    out = (f"{head}\nRevisions in range: {total}{'+ (stopped at the page cap)' if capped else ''} "
           f"(distinct editors: {len(users)}; counts every revision, including minor and bot edits)\n"
           f"First: {first[0]} by {first[1]}\nLast: {last[0]} by {last[1]}")
    return out + ("\n" + "\n".join(listing) if listing else "")


@tool
def wikipedia_search(query: str, limit: int = 5, namespace: str = "") -> str:
    """Find English Wikipedia pages by words. Returns 'title | snippet | size | last edited' lines.
    namespace: '' (articles, default), 'talk', 'wikipedia' (project pages such as featured-article logs
    and candidate archives), 'category', 'template', 'draft', 'portal' or 'all'.
    Tip: search syntax like intitle:"featured log" works. Then read a page with wikipedia_lookup."""
    q = " ".join(str(query).split())
    if not q:
        return "wikipedia_search error: empty query"
    ns = _NS.get(str(namespace).strip().lower())
    if ns is None:
        return f"wikipedia_search error: unknown namespace '{namespace}'. Use one of: " + ", ".join(k or "''" for k in _NS)
    try:
        lim = max(1, min(int(limit), 10))
    except (TypeError, ValueError):
        lim = 5
    key = (q.lower(), lim, ns)
    with _lock:
        hit = _searches.get(key)
    if hit and time.time() - hit[0] < _CACHE_TTL:
        return hit[1]

    def run():
        hits = _api(action="query", list="search", srsearch=q, srlimit=lim, srnamespace=ns,
                    srprop="snippet|wordcount|timestamp")["query"]["search"]
        if not hits:
            return "No Wikipedia results. Try fewer or different words."
        lines = []
        for i, h in enumerate(hits, 1):
            snip = " ".join(htmlmod.unescape(re.sub(r"<[^>]+>", "", h.get("snippet", ""))).split())[:200]
            lines.append(f"{i}. {h['title']} | {snip} | {h.get('wordcount', '?')} words | edited {h.get('timestamp', '')[:10]}")
        text = "\n".join(lines)
        with _lock:
            _searches[key] = (time.time(), text)
        return text

    return _guard("wikipedia_search", run)


@tool
def wikipedia_lookup(title: str, as_of: str = "", section: str = "", offset: int = 0) -> str:
    """Read an English Wikipedia page: text, infobox and tables included. Also works for Talk pages and
    project pages, e.g. 'Talk:Giganotosaurus' or 'Wikipedia:Featured article candidates/...'.

    Args:
        title: Page title or search words, e.g. "Mercedes Sosa".
        as_of: Optional "YYYY", "YYYY-MM" or "YYYY-MM-DD". Returns the page as it was at the end of that
            period. For "the latest 2022 version" pass "2022". Empty = current version.
        section: Optional section name, e.g. "Discography". Returns only that section and its subsections
            ("Lead" gives the introduction). The first page of an article lists all section names.
        offset: Character offset for continuing long output; use the value from the footer.

    If the footer says more is available, call again with the same arguments and the offset it gives."""
    try:
        off = max(0, int(offset))
    except (TypeError, ValueError):
        off = 0
    return _guard("wikipedia_lookup", lambda: _lookup(title, as_of, section, off))


@tool
def wikipedia_history(title: str, start: str = "", end: str = "", list_limit: int = 0) -> str:
    """Count (and optionally list) the edits (revisions) of a Wikipedia page between two dates, inclusive.
    Use it for 'how many edits ...' questions.

    Args:
        title: Page title, e.g. "Antidisestablishmentarianism".
        start: Optional "YYYY", "YYYY-MM" or "YYYY-MM-DD" (UTC). Empty = from page creation.
        end: Optional date in the same formats; the whole end day (or month/year) is included. For
            'until June 2023' decide whether you mean the start or end of June and pass 2023-05-31 or 2023-06-30.
        list_limit: Also list the first N edits in the range, oldest first (default 0 = just the count)."""
    try:
        lim = max(0, min(int(list_limit), 50))
    except (TypeError, ValueError):
        lim = 0
    return _guard("wikipedia_history", lambda: _history(title, start, end, lim))