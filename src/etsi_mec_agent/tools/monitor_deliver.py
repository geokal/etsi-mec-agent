import os
import json
import re
import time
import urllib.request
from pathlib import Path
from typing import List, Dict

from .exa_search import find_etsi_pdf_url

# Constants
MANIFEST_FILE = Path(__file__).parent.parent.parent.parent / "data" / "specs" / "manifest.json"
DATA_SPEC_DIR = Path(__file__).parent.parent.parent.parent / "data" / "specs"

def load_manifest() -> Dict[str, str]:
    """Load the manifest mapping spec IDs to the latest known PDF URL or version.
    Returns an empty dict if the file does not exist.
    """
    if MANIFEST_FILE.is_file():
        try:
            with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_manifest(manifest: Dict[str, str]):
    """Persist the manifest to disk atomically."""
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    temp_path = MANIFEST_FILE.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    temp_path.replace(MANIFEST_FILE)

def _download_pdf(url: str, target_path: Path):
    """Download a PDF from *url* into *target_path*.
    Includes a User‑Agent header to avoid 403 errors.
    Raises on HTTP errors.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)
    # Add a common browser User‑Agent header
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        if resp.status != 200:
            raise RuntimeError(f"Failed to download {url}: HTTP {resp.status}")
        head = resp.read(4)
        if head != b"%PDF":
            raise RuntimeError(f"{url} is not a PDF (starts with {head!r})")
        with open(target_path, "wb") as out:
            out.write(head)
            while True:
                chunk = resp.read(65536)   # 64 KB at a time — never buffers full PDF in RAM
                if not chunk:
                    break
                out.write(chunk)

def _spec_id_from_url(url: str) -> str:
    """Return the spec ID declared by the URL's own filename.

        .../002/04.01.01_60/gs_MEC002v040101p.pdf      -> MEC002
        .../025/02.01.01_60/gr_mec-dec025v020101p.pdf  -> MEC-DEC025

    The directory component is deliberately not used: for redirected or withdrawn
    documents it disagrees with the file ETSI actually serves.
    """
    stem = url.rstrip("/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
    m = re.match(r"[a-z]+_([a-z-]+?)(\d{3,5})v\d{4,6}[a-z]?$", stem, re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot infer spec ID from {url}")
    num = m.group(2)
    if len(num) == 5:                      # gs_mec01002… is ETSI series part 010-02
        num = f"{num[:3]}-{num[3:]}"
    return f"{m.group(1).upper()}{num}"

def _version_from_url(url: str) -> tuple:
    """Extract version tuple (major, minor, patch) from an ETSI deliver URL.
    The URL contains a segment like 'v040101' where the first two digits are major,
    next two minor, next two patch. Returns (major, minor, patch) as ints.
    If the pattern is not found, returns (0, 0, 0) as a fallback.
    """
    m = re.search(r"v(\d{2})(\d{2})(\d{2})", url, re.IGNORECASE)
    if m:
        major, minor, patch = map(int, m.groups())
        return (major, minor, patch)
    return (0, 0, 0)

# Updated monitor loop with version comparison
def monitor_etsi_deliver(poll_interval: int = 3600) -> None:
    """Continuously poll the ETSI deliver directory for new PDF releases.
    Only updates when a higher version is found.
    """
    manifest = load_manifest()
    while True:
        # Build poll list from known specs + a 3-spec look-ahead for newly published ones.
        # This reduces 99 Exa calls/cycle down to ~len(manifest)+3 calls.
        known_nums = [int(k[3:]) for k in manifest if k[3:].isdigit()]
        max_num = max(known_nums, default=0)
        look_ahead = [f"MEC{i:03d}" for i in range(max_num + 1, max_num + 4)]
        spec_ids = sorted(set(manifest.keys()) | set(look_ahead))
        if not spec_ids:
            spec_ids = [f"MEC{i:03d}" for i in range(1, 10)]   # bootstrap: try first 9
        updated = False
        for spec_id in spec_ids:
            query = f"{spec_id} site:etsi.org deliver pdf"
            try:
                pdf_url = find_etsi_pdf_url(query)
            except Exception as exc:
                print(f"[monitor] error locating PDF for {spec_id}: {exc}")
                continue
            if not pdf_url:
                continue

            # A search for an unpublished or withdrawn spec returns the nearest
            # popular PDF, so the URL's own number must match the key before saving.
            try:
                found_id = _spec_id_from_url(pdf_url)
            except ValueError as exc:
                print(f"[monitor] skipping {spec_id}: {exc}")
                continue
            # Store under the identity the document declares, not the number we
            # asked for: a lookup of an unpublished spec returns a neighbour PDF,
            # and naming it after the request is how 52 duplicates were created.
            if found_id != spec_id:
                print(f"[monitor] {spec_id}: lookup returned {found_id} - filing as {found_id}.pdf")

            # URL we previously recorded for this spec (if any)
            known_url = manifest.get(found_id)

            # Determine the highest known version from existing local PDFs (which may have version suffixes)
            # Look for files like MEC001_v3.2.1.pdf or gs_MEC001v030201p.pdf etc.
            existing_files = list(DATA_SPEC_DIR.glob(f"{found_id}*pdf"))
            known_version = (0, 0, 0)
            if existing_files:
                for f in existing_files:
                    m = re.search(r"v(\d{2})(\d{2})(\d{2})", f.name, re.IGNORECASE)
                    if m:
                        v = tuple(map(int, m.groups()))
                        if v > known_version:
                            known_version = v
                if known_url:
                    manifest_version = _version_from_url(known_url)
                    if manifest_version > known_version:
                        known_version = manifest_version
            else:
                if known_url:
                    known_version = _version_from_url(known_url)

            # Local PDF is stored directly under data/specs (no extra sub‑folder)
            local_path = DATA_SPEC_DIR / f"{found_id}.pdf"

            # Version from the freshly discovered URL
            new_version = _version_from_url(pdf_url)

            # Download only when the newly found version is newer
            if new_version > known_version:
                print(f"[monitor] new version for {spec_id}: {pdf_url} (v{new_version[0]}.{new_version[1]}.{new_version[2]})")
                try:
                    _download_pdf(pdf_url, local_path)
                except Exception as exc:
                    print(f"[monitor] failed to download {pdf_url}: {exc}")
                    continue
                manifest[found_id] = pdf_url
                updated = True
        if updated:
            save_manifest(manifest)
            print("[monitor] manifest updated")
        else:
            print("[monitor] no new PDFs detected")
        time.sleep(poll_interval)

# Haystack‑compatible wrapper (optional)
try:
    from haystack import Tool
except Exception:
    # Define a no‑op decorator if Haystack is missing
    def Tool(*_, **__):
        def decorator(func):
            return func
        return decorator

@Tool(name="monitor_etsi_deliver", description="Continuously monitor the ETSI deliver directory for new PDF releases and download them. Use with caution – it runs forever until stopped. Provide poll_interval in seconds.")
def monitor_etsi_deliver_tool(poll_interval: int = 3600) -> str:
    """Entry point used by Haystack. Starts the monitor in a background thread.
    Returns a short status message; the actual monitoring runs until the
    process is terminated.
    """
    import threading
    thread = threading.Thread(target=monitor_etsi_deliver, args=(poll_interval,), daemon=True)
    thread.start()
    return f"Started ETSI deliver monitor with {poll_interval}s interval (background thread)."
