"""AVeriTeC claims, page filtering, passages and per-claim BM25 search.

The dev knowledge store is read remotely one claim at a time (no full download).
Each claim's passages go into an SQLite FTS5 index that lives only while that
claim runs.
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator
from urllib.parse import urlparse

STORE_URL = (
    "https://huggingface.co/chenxwh/AVeriTeC/resolve/main/"
    "data_store/knowledge_store/dev_knowledge_store.zip"
)
DEV_URL = "https://huggingface.co/chenxwh/AVeriTeC/resolve/main/data/dev.json"
SITES_FILE = Path(__file__).with_name("factcheck_sites.txt")
SENTENCES_PER_PASSAGE = 5
TOP_K = 5
FACTCHECK_PATH = re.compile(r"fact[-_ ]?check", re.I)
ARCHIVE_PREFIX = re.compile(r"^https?://web\.archive\.org/web/[^/]+/", re.I)
LABEL_TO_SIDE = {"Supported": "true", "Refuted": "false"}


@dataclass(frozen=True)
class GoldQuestion:
    question: str
    urls: tuple[str, ...]


@dataclass(frozen=True)
class Claim:
    claim_id: str
    text: str
    label: str
    side: str  # "true" or "false": the correct side
    fact_check_url: str
    reannotated: bool
    questions: tuple[GoldQuestion, ...]


@dataclass(frozen=True)
class Passage:
    pid: str
    url: str
    is_gold: bool
    text: str


@dataclass(frozen=True)
class Page:
    url: str
    is_gold: bool
    sentences: tuple[str, ...] = field(default_factory=tuple)


def load_sites(path: Path = SITES_FILE) -> frozenset[str]:
    lines = path.read_text().splitlines()
    return frozenset(l.strip().lower() for l in lines if l.strip() and not l.startswith("#"))


def unwrap(url: str) -> str:
    return ARCHIVE_PREFIX.sub("", url or "")


def host_of(url: str) -> str:
    target = unwrap(url)
    if "://" not in target:
        target = "http://" + target
    host = urlparse(target).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def norm_url(url: str) -> str:
    """Unwrapped URL without scheme, leading www. or trailing slash, lower-cased host."""
    target = unwrap(url).strip()
    target = re.sub(r"^https?://", "", target, flags=re.I)
    host, _, rest = target.partition("/")
    host = host.lower()
    host = host[4:] if host.startswith("www.") else host
    return (host + ("/" + rest if rest else "")).rstrip("/")


def _site_match(nurl: str, site: str) -> bool:
    host, _, path = site.partition("/")
    url_host, _, url_path = nurl.partition("/")
    host_ok = url_host == host or url_host.endswith("." + host)
    return host_ok and (not path or url_path.lower().startswith(path.lower()))


def is_factcheck(url: str, own_article: str, sites: frozenset[str]) -> bool:
    """True if the page is the claim's own fact-check, on a listed fact-check site
    (host, or host/path prefix), or has "fact check" in its path."""
    nurl = norm_url(url)
    if own_article and norm_url(own_article) == nurl:
        return True
    if any(_site_match(nurl, s) for s in sites):
        return True
    return bool(FACTCHECK_PATH.search(nurl.partition("/")[2]))


def parse_claims(raw: list[dict]) -> list[Claim]:
    claims = []
    for idx, row in enumerate(raw):
        if row.get("label") not in LABEL_TO_SIDE:
            continue
        questions = tuple(
            GoldQuestion(
                question=q["question"],
                urls=tuple(a.get("source_url", "") for a in q.get("answers", []) if a.get("source_url")),
            )
            for q in row.get("questions", [])
        )
        claims.append(
            Claim(
                claim_id=str(idx),
                text=row["claim"],
                label=row["label"],
                side=LABEL_TO_SIDE[row["label"]],
                fact_check_url=row.get("fact_checking_article") or "",
                reannotated=bool(row.get("required_reannotation")),
                questions=questions,
            )
        )
    return claims


def load_claims(path: Path) -> list[Claim]:
    return parse_claims(json.loads(Path(path).read_text()))


def parse_store_lines(lines: Iterable[str]) -> list[Page]:
    pages = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        sentences = tuple(s.strip() for s in row.get("url2text", []) if s and s.strip())
        pages.append(Page(url=row.get("url", ""), is_gold=row.get("type") == "gold", sentences=sentences))
    return pages


def filter_pages(pages: list[Page], claim: Claim, sites: frozenset[str]) -> list[Page]:
    return [p for p in pages if not is_factcheck(p.url, claim.fact_check_url, sites)]


def make_passages(pages: list[Page], size: int = SENTENCES_PER_PASSAGE) -> list[Passage]:
    passages = []
    for page_no, page in enumerate(pages):
        for start in range(0, len(page.sentences), size):
            chunk = " ".join(page.sentences[start : start + size])
            if chunk:
                passages.append(Passage(f"p{page_no}-{start // size}", page.url, page.is_gold, chunk))
    return passages


class SearchIndex:
    """BM25 over one claim's passages, in an SQLite FTS5 table."""

    def __init__(self, passages: list[Passage], db_path: str = ":memory:", check_same_thread: bool = True):
        self._conn = sqlite3.connect(db_path, check_same_thread=check_same_thread)
        self._lock = threading.Lock()
        self._conn.execute("CREATE VIRTUAL TABLE p USING fts5(pid UNINDEXED, body)")
        self._conn.executemany("INSERT INTO p VALUES (?, ?)", [(x.pid, x.text) for x in passages])
        self._by_id = {x.pid: x for x in passages}

    @staticmethod
    def to_match(query: str) -> str:
        terms = re.findall(r"[A-Za-z0-9]+", query.lower())
        return " OR ".join(f'"{t}"' for t in terms if len(t) > 1)

    def search(self, query: str, k: int = TOP_K) -> list[Passage]:
        match = self.to_match(query)
        if not match:
            return []
        with self._lock:
            rows = self._conn.execute(
                "SELECT pid FROM p WHERE p MATCH ? ORDER BY bm25(p) LIMIT ?", (match, k)
            ).fetchall()
        return [self._by_id[r[0]] for r in rows]

    def get(self, pid: str) -> Passage | None:
        return self._by_id.get(pid)

    def close(self) -> None:
        self._conn.close()


def iter_remote_claim_lines(claim_ids: Iterable[str], url: str = STORE_URL) -> Iterator[tuple[str, list[str]]]:
    """Yield (claim_id, jsonl lines) for each claim, read remotely from the zip."""
    from remotezip import RemoteZip

    with RemoteZip(url) as archive:
        names = set(archive.namelist())
        for cid in claim_ids:
            name = f"output_dev/{cid}.json"
            if name not in names:
                yield cid, []
                continue
            yield cid, archive.read(name).decode("utf-8", errors="replace").split("\n")
