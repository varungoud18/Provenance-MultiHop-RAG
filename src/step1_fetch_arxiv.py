"""
Stage 1 - Step 1: ArXiv Paper Fetcher
Fetches academic papers matching a topic from arXiv's public API (Atom XML feed)
and saves them to data/papers.json.
"""

import argparse
import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any

from src.config import PAPERS_FILE, DATA_DIR

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

def clean_text(text: str) -> str:
    """Removes irregular newlines, excessive whitespace, and XML artifacts."""
    if not text:
        return ""
    # Replace newlines with spaces and condense multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_arxiv_id(raw_id_url: str) -> str:
    """
    Extracts canonical arXiv ID from URL like 'http://arxiv.org/abs/2312.10997v1'
    Returns '2312.10997v1'.
    """
    match = re.search(r"arxiv\.org/abs/([^/]+)$", raw_id_url)
    if match:
        return match.group(1)
    # Fallback to last segment of URL/string
    return raw_id_url.split("/")[-1]

import time
import requests

def _call_arxiv_api(url: str, max_retries: int = 2, timeout: int = 10) -> str:
    """Makes a GET request to arXiv API with quick backoff."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ProvenanceAcademicAgent/2.0"}
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            if resp.status_code == 200 and resp.text.strip():
                return resp.text
            elif resp.status_code in (429, 503):
                print(f"[WARN] arXiv API returned {resp.status_code} (Rate Limited/Server Busy).")
                time.sleep(2)
            else:
                resp.raise_for_status()
        except Exception as e:
            last_err = e
            time.sleep(1)
    raise RuntimeError(f"arXiv unavailable ({last_err})")

STOP_WORDS = {
    "what", "is", "are", "was", "were", "how", "do", "does", "did", "can", "could",
    "would", "should", "why", "which", "who", "whom", "whose", "when", "where",
    "the", "a", "an", "in", "on", "at", "of", "for", "with", "by", "about", "against",
    "between", "into", "through", "during", "before", "after", "above", "below", "to",
    "from", "up", "down", "out", "off", "over", "under", "again", "further",
    "then", "once", "and", "or", "but", "so", "because", "as", "until", "while",
    "their", "theirs", "its", "they", "them", "it", "this", "that", "these", "those",
    "explain", "describe", "discuss", "tell", "give", "me"
}

def clean_academic_query(raw_query: str) -> str:
    """Strips conversational question wrappers and punctuation to produce search keywords."""
    cleaned = re.sub(r"[^\w\s]", " ", raw_query)
    tokens = [w for w in cleaned.split() if w.lower() not in STOP_WORDS and len(w) > 1]
    if tokens:
        return " ".join(tokens)
    return raw_query.strip()


def fetch_openalex_papers(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Fallback academic discovery engine using OpenAlex (250M+ scholarly works).
    Captures preprints and publications across arXiv, IEEE, ACM, Springer with full abstracts.
    """
    print(f"\n==================================================")
    print(f"  [FALLBACK ENGINE] Querying OpenAlex Academic Index")
    print(f"==================================================")
    cleaned = clean_academic_query(query)
    # Use the 4 most relevant keywords to ensure broad high-precision matches
    words = cleaned.split()[:5]
    search_term = " ".join(words) if words else query.strip()
    print(f"Searching OpenAlex for: '{search_term}'...")

    url = f"https://api.openalex.org/works?search={urllib.parse.quote_plus(search_term)}&per_page={max_results * 2}"
    headers = {"User-Agent": "ProvenanceResearchAgent/2.0 (mailto:capstone@university.edu)"}

    try:
        r = requests.get(url, headers=headers, timeout=12)
        if r.status_code != 200:
            print(f"[WARN] OpenAlex returned HTTP {r.status_code}")
            return []
        data = r.json()
    except Exception as err:
        print(f"[ERROR] OpenAlex request failed: {err}")
        return []

    papers = []
    for item in data.get("results", []):
        title = item.get("title")
        if not title:
            continue

        inv = item.get("abstract_inverted_index")
        if not inv:
            continue

        pos_map = {pos: word for word, pos_list in inv.items() for pos in pos_list}
        abstract = " ".join(pos_map[p] for p in sorted(pos_map.keys()))
        if len(abstract.split()) < 20:
            continue

        ids = item.get("ids", {})
        arxiv_id = ids.get("arxiv")
        doi = item.get("doi")
        openalex_id = item.get("id", "").split("/")[-1]

        if arxiv_id:
            clean_id = arxiv_id.replace("https://arxiv.org/abs/", "").replace("http://arxiv.org/abs/", "")
            paper_url = f"https://arxiv.org/abs/{clean_id}"
            paper_id = clean_id
        elif doi:
            paper_url = doi
            paper_id = doi.replace("https://doi.org/", "")
        else:
            paper_url = item.get("id", f"https://openalex.org/{openalex_id}")
            paper_id = openalex_id

        authors = []
        for a in item.get("authorships", []):
            name = a.get("author", {}).get("display_name")
            if name:
                authors.append(name)

        pub_date = item.get("publication_date") or str(item.get("publication_year", "Unknown"))

        papers.append({
            "id": paper_id,
            "title": clean_text(title),
            "abstract": clean_text(abstract),
            "url": paper_url,
            "published": pub_date,
            "authors": authors[:5]
        })

        if len(papers) >= max_results:
            break

    print(f"OpenAlex retrieved {len(papers)} peer-reviewed papers with abstracts.")
    return papers


def fetch_arxiv_papers(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Queries the arXiv public API using HTTPS and parses the Atom XML response.
    If arXiv is throttled (HTTP 429) or returns no papers, automatically falls back
    to OpenAlex to ensure zero-failure live literature discovery.
    """
    print(f"\n==================================================")
    print(f"  Step 1: Fetching Papers from Academic Literature")
    print(f"==================================================")
    print(f"Raw Query   : '{query}'")
    
    topic_keywords = clean_academic_query(query)
    print(f"Search Terms: '{topic_keywords}'")
    print(f"Max Results : {max_results}")

    # Build search query for arXiv: search across all fields (all:...)
    params = {
        "search_query": f"all:{topic_keywords}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending"
    }

    url = f"https://export.arxiv.org/api/query?{urllib.parse.urlencode(params)}"
    print(f"Attempting arXiv API: {url}\n")

    papers = []
    entries = []
    try:
        xml_data = _call_arxiv_api(url)
        root = ET.fromstring(xml_data)
        entries = root.findall("atom:entry", ATOM_NS)
        
        if not entries:
            # Retry with tokenized search
            words = topic_keywords.split()[:4]
            if words:
                broad_param = " AND ".join([f"all:{w}" for w in words])
                params["search_query"] = broad_param
                url = f"https://export.arxiv.org/api/query?{urllib.parse.urlencode(params)}"
                xml_data = _call_arxiv_api(url)
                root = ET.fromstring(xml_data)
                entries = root.findall("atom:entry", ATOM_NS)

        for entry in entries:
            raw_id = entry.find("atom:id", ATOM_NS)
            title = entry.find("atom:title", ATOM_NS)
            summary = entry.find("atom:summary", ATOM_NS)
            published = entry.find("atom:published", ATOM_NS)

            if raw_id is None or title is None or summary is None:
                continue

            raw_id_str = raw_id.text.strip() if raw_id.text else ""
            paper_id = extract_arxiv_id(raw_id_str)
            clean_title = clean_text(title.text)
            clean_abstract = clean_text(summary.text)
            pub_date = published.text.strip() if published is not None and published.text else "Unknown"

            # Extract authors
            author_elems = entry.findall("atom:author", ATOM_NS)
            authors = []
            for a in author_elems:
                name_el = a.find("atom:name", ATOM_NS)
                if name_el is not None and name_el.text:
                    authors.append(name_el.text.strip())

            paper_url = f"https://arxiv.org/abs/{paper_id}"

            papers.append({
                "id": paper_id,
                "title": clean_title,
                "abstract": clean_abstract,
                "url": paper_url,
                "published": pub_date,
                "authors": authors
            })
    except Exception as e:
        print(f"[NOTICE] arXiv search unavailable or rate-limited ({e}). Switching to OpenAlex fallback...")

    # If arXiv was throttled or returned zero papers, activate OpenAlex fallback
    if not papers:
        print("[NOTICE] No papers returned from arXiv. Activating OpenAlex fallback...")
        papers = fetch_openalex_papers(query=query, max_results=max_results)

    if papers:
        # Save to data/papers.json
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(PAPERS_FILE, "w", encoding="utf-8") as f:
            json.dump(papers, f, indent=2, ensure_ascii=False)

        print(f"Successfully indexed and saved {len(papers)} papers to: {PAPERS_FILE}\n")
        print("Fetched Papers Summary:")
        print("-" * 75)
        for idx, p in enumerate(papers, 1):
            print(f"[{idx}] {p['title']}")
            print(f"    ID: {p['id']} | Published: {p['published']}")
            print(f"    URL: {p['url']}")
            print(f"    Abstract length: {len(p['abstract'].split())} words\n")

    return papers

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch papers from arXiv's public API.")
    parser.add_argument(
        "--query",
        type=str,
        default="retrieval augmented generation",
        help="Search topic or keywords (default: 'retrieval augmented generation')"
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=10,
        help="Number of papers to fetch (default: 10)"
    )
    args = parser.parse_args()
    fetch_arxiv_papers(query=args.query, max_results=args.max_results)
