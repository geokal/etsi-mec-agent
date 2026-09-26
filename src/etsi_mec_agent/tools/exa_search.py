import os
import json
import urllib.request
import urllib.parse
from typing import Optional, List, Dict

# Exa API base endpoint
EXA_API_URL = "https://api.exa.ai/search"

def _get_api_key() -> str:
    """Read the Exa API key from the environment.
    Raises RuntimeError if not set.
    """
    key = os.getenv("EXA_API_KEY")
    if not key:
        raise RuntimeError("EXA_API_KEY environment variable not set")
    return key

def _make_request(query: str, num_results: int = 5) -> Dict:
    """Perform a POST request to the Exa search endpoint.
    Exa API expects a JSON payload with the query and number of results.
    Returns the parsed JSON response.
    """
    payload = json.dumps({"query": query, "num_results": num_results}).encode('utf-8')
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {_get_api_key()}",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }
    req = urllib.request.Request(EXA_API_URL, data=payload, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        if resp.status != 200:
            raise RuntimeError(f"Exa search failed: HTTP {resp.status}")
        return json.load(resp)

def search_etsi_web(query: str, num_results: int = 5) -> List[Dict]:
    """Search the web (restricted to ETSI domains) via Exa.
    Returns a list of result dictionaries containing at least `title`, `url`, and `snippet`.
    """
    etsified = f"{query} site:etsi.org"
    resp = _make_request(etsified, num_results=num_results)
    return resp.get("results", [])

def find_etsi_pdf_url(query: str) -> Optional[str]:
    """Return the first PDF URL from Exa results that points to the official ETSI deliver directory.
    The function looks for URLs ending with `.pdf` and containing the substring `/deliver/`.
    If no such URL is found, returns ``None``.
    """
    results = search_etsi_web(query, num_results=10)
    for result in results:
        url = result.get("url", "")
        if url.lower().endswith('.pdf') and "/deliver/" in url.lower():
            return url
    return None
