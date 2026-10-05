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

def _call_arxiv_api(url: str, max_retries: int = 3, timeout: int = 25) -> str:
    """Makes a resilient GET request to arXiv API with retries and exponential backoff."""
    headers = {"User-Agent": "AcademicResearchAgent/1.0 (Capstone Project; mailto:student@university.edu)"}
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            if resp.status_code == 200 and resp.text.strip():
                return resp.text
            elif resp.status_code == 503:
                # arXiv rate limit or temporary server busy
                wait_sec = attempt * 3
                print(f"[WARN] arXiv API returned 503 (Server Busy). Retrying in {wait_sec}s (attempt {attempt}/{max_retries})...")
                time.sleep(wait_sec)
            else:
                resp.raise_for_status()
        except Exception as e:
            last_err = e
            wait_sec = attempt * 2
            print(f"[WARN] arXiv query attempt {attempt}/{max_retries} failed ({e}). Retrying in {wait_sec}s...")
            time.sleep(wait_sec)
    raise RuntimeError(f"Failed to reach arXiv API after {max_retries} attempts: {last_err}")

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

def fetch_arxiv_papers(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Queries the arXiv public API using HTTPS and parses the Atom XML response.
    Resilient with retries, smart keyword extraction, and broad fallback. No API key required.
    """
    print(f"\n==================================================")
    print(f"  Step 1: Fetching Papers from arXiv API")
    print(f"==================================================")
    print(f"Raw Query   : '{query}'")
    
    topic_keywords = clean_academic_query(query)
    print(f"Search Terms: '{topic_keywords}'")
    print(f"Max Results : {max_results}")

    # Build search query: search across all fields (all:...)
    params = {
        "search_query": f"all:{topic_keywords}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending"
    }

    url = f"https://export.arxiv.org/api/query?{urllib.parse.urlencode(params)}"
    print(f"Calling arXiv API: {url}\n")

    try:
        xml_data = _call_arxiv_api(url)
    except Exception as e:
        print(f"[ERROR] Failed to query arXiv API: {e}")
        raise

    root = ET.fromstring(xml_data)
    entries = root.findall("atom:entry", ATOM_NS)
    
    if not entries:
        # Retry with quoted phrase
        print("[NOTICE] No results for unquoted search. Retrying with quoted phrase...")
        params["search_query"] = f'all:"{topic_keywords}"'
        url = f"https://export.arxiv.org/api/query?{urllib.parse.urlencode(params)}"
        xml_data = _call_arxiv_api(url)
        root = ET.fromstring(xml_data)
        entries = root.findall("atom:entry", ATOM_NS)

    if not entries:
        # Retry with broad AND tokens
        print("[NOTICE] Retrying with tokenized search...")
        words = topic_keywords.split()[:4]
        broad_param = " AND ".join([f"all:{w}" for w in words])
        params["search_query"] = broad_param
        url = f"https://export.arxiv.org/api/query?{urllib.parse.urlencode(params)}"
        xml_data = _call_arxiv_api(url)
        root = ET.fromstring(xml_data)
        entries = root.findall("atom:entry", ATOM_NS)

    papers = []
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

        # Guaranteed working URL
        paper_url = f"https://arxiv.org/abs/{paper_id}"

        papers.append({
            "id": paper_id,
            "title": clean_title,
            "abstract": clean_abstract,
            "url": paper_url,
            "published": pub_date,
            "authors": authors
        })

    # Save to data/papers.json
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(PAPERS_FILE, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

    print(f"Successfully fetched and saved {len(papers)} papers to: {PAPERS_FILE}\n")
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
