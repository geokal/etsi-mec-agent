import os
import json
import urllib.request
from pathlib import Path
from typing import List, Dict

# Constants
DATA_SPEC_DIR = Path(__file__).parent.parent / "data" / "specs"
BASE_API = "https://forge.etsi.org/rep/api/v4"
BASE_RAW = "https://forge.etsi.org/rep"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

class ETSIForgeClient:
    """Simple client for ETSI Forge (GitLab) public API.
    Provides methods to list specification projects under the "specifications"
    subgroup and download markdown files.
    """
    def _get(self, url: str) -> any:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status != 200:
                raise RuntimeError(f"GET {url} failed: HTTP {resp.status}")
            return json.load(resp)

    def _list_subgroups(self) -> List[Dict]:
        return self._get(f"{BASE_API}/groups/mec/subgroups")

    def _list_projects(self, subgroup_id: int) -> List[Dict]:
        return self._get(f"{BASE_API}/groups/{subgroup_id}/projects")

    def list_spec_projects(self) -> List[Dict]:
        """Return a list of project dicts within the "specifications" subgroup.
        Each dict contains at least ``id``, ``name`` and ``path``.
        """
        for sub in self._list_subgroups():
            if sub.get("path", "").endswith("/specifications"):
                return self._list_projects(sub["id"])
        return []

    def list_md_files(self, project_path: str) -> List[str]:
        """List markdown file paths (relative) in the given project.
        Uses the repository tree endpoint; filters for *.md blobs.
        """
        enc_path = urllib.parse.quote_plus(project_path)
        url = f"{BASE_API}/projects/{enc_path}/repository/tree?recursive=true"
        entries = self._get(url)
        md_files = [e["path"] for e in entries if e.get("type") == "blob" and e["path"].lower().endswith('.md')]
        return md_files

    def download_raw_file(self, project_path: str, file_path: str, target_path: Path) -> None:
        """Download a raw file from a project.
        Uses the simple raw URL pattern: /rep/<project_path>/raw/master/<file_path>
        """
        raw_url = f"{BASE_RAW}/{project_path}/raw/master/{file_path}"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(raw_url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Download {raw_url} failed: HTTP {resp.status}")
            data = resp.read()
            with open(target_path, "wb") as out:
                out.write(data)

# Haystack‑compatible wrapper ---------------------------------------------------
try:
    from haystack import Tool
except Exception:
    # Define a no‑op decorator if Haystack is missing
    def Tool(*_, **__):
        def decorator(func):
            return func
        return decorator

@Tool(name="download_specifications_markdown", description="Download all markdown specification files from the ETSI Forge 'specifications' subgroup into the local data directory (data/specs/markdown). Returns a short status message.")
def download_specifications_markdown_tool() -> str:
    client = ETSIForgeClient()
    projects = client.list_spec_projects()
    count = 0
    for proj in projects:
        proj_path = proj["path"]
        proj_name = proj["name"]
        md_files = client.list_md_files(proj_path)
        for md in md_files:
            target = DATA_SPEC_DIR / "markdown" / proj_name / md
            client.download_raw_file(proj_path, md, target)
            count += 1
    return f"Downloaded {count} markdown files from {len(projects)} ETSI Forge specification projects."
