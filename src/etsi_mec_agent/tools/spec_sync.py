import argparse
import os
import re
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

try:
    from haystack.tools import Tool as _HaystackTool   # optional — only needed if haystack-ai is installed
except ImportError:
    _HaystackTool = None  # type: ignore

from etsi_mec_agent.tools.etsi_forge import ETSIForgeClient, forge_client
from etsi_mec_agent.tools.exa_search import find_etsi_pdf_url

# Standard registry of ETSI MEC specifications and corresponding Forge repos
MEC_SPEC_CATALOG = {
    "MEC002": {"name": "Technical Requirements", "forge_repo": None, "type": "GS"},
    "MEC003": {"name": "Framework and Reference Architecture", "forge_repo": None, "type": "GS"},
    "MEC010-2": {"name": "Application Lifecycle, Rules and Requirements", "forge_repo": None, "type": "GS"},
    "MEC011": {"name": "Edge Platform Application Enablement", "forge_repo": "gs011-app-enablement-api", "type": "GS"},
    "MEC012": {"name": "Radio Network Information API", "forge_repo": "gs012-rnis-api", "type": "GS"},
    "MEC013": {"name": "Location API", "forge_repo": "gs013-location-api", "type": "GS"},
    "MEC014": {"name": "UE Identity API", "forge_repo": "gs014-ue-identity-api", "type": "GS"},
    "MEC015": {"name": "Bandwidth Management API", "forge_repo": "gs015-bw-mgmt-api", "type": "GS"},
    "MEC016": {"name": "Device Application Interface", "forge_repo": "gs016-dev-app-api", "type": "GS"},
    "MEC021": {"name": "Application Mobility Service API", "forge_repo": "gs021-ams-api", "type": "GS"},
    "MEC027": {"name": "Operator Platform Engagement", "forge_repo": None, "type": "GS"},
    "MEC028": {"name": "WLAN Information API", "forge_repo": "gs028-wai-api", "type": "GS"},
    "MEC029": {"name": "Fixed Access Information API", "forge_repo": "gs029-fai-api", "type": "GS"},
    "MEC030": {"name": "V2X Information Service API", "forge_repo": "gs030-v2x-api", "type": "GS"},
    "MEC031": {"name": "MEC 5G Integration", "forge_repo": None, "type": "GR"},
    "MEC035": {"name": "Inter-MEC system communication", "forge_repo": None, "type": "GR"},
    "MEC037": {"name": "Abstract Test Suite (ATS)", "forge_repo": "gs037-ats", "type": "GS"},
    "MEC040": {"name": "Federation Enablement API", "forge_repo": "gs040-fed-api", "type": "GS"},
}


def normalize_spec_id(raw: str) -> str:
    """Standardize spec IDs to 'MEC011', 'MEC003', 'MEC010-2', etc."""
    raw = raw.upper().strip()
    match = re.search(r"(?:MEC|GS|GR)?[_\-\s]?0*([0-9]{1,3})(-[0-9]+)?", raw)
    if match:
        num = int(match.group(1))
        suffix = match.group(2) or ""
        return f"MEC{num:03d}{suffix}"
    return raw


def scan_local_specs(specs_dir: Path) -> Dict[str, List[Path]]:
    """
    Scan local directory for existing ETSI MEC specifications.
    Returns a dictionary mapping detected Spec ID (e.g. 'MEC011') to existing local file paths.
    """
    found: Dict[str, List[Path]] = {}
    if not specs_dir.exists():
        return found

    for file_path in specs_dir.rglob("*"):
        if not file_path.is_file() or file_path.suffix.lower() not in [".pdf", ".yaml", ".json"]:
            continue

        full_rel_str = str(file_path.relative_to(specs_dir)).lower()

        # Match patterns like gs_mec011, mec011, gs012, mec-011 in filename or path
        match = re.search(r"(?:mec|gs|gr)[_\-\s]?0*([0-9]{1,3}(?:-[0-9]+)?)", full_rel_str)
        if match:
            spec_key = normalize_spec_id(match.group(1))
            found.setdefault(spec_key, []).append(file_path)
            continue

        # Check catalog titles & forge repos
        for spec_key, meta in MEC_SPEC_CATALOG.items():
            repo = meta.get("forge_repo") or ""
            if repo and repo.lower() in full_rel_str:
                found.setdefault(spec_key, []).append(file_path)
                break

            words = meta["name"].lower().split()
            if sum(1 for w in words if len(w) > 3 and w in full_rel_str) >= 2:
                found.setdefault(spec_key, []).append(file_path)
                break

    return found


def download_direct_pdf(url: str, destination: Path) -> bool:
    """Download a PDF directly from an HTTP URL."""
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
            destination.write_bytes(data)
        return True
    except Exception as e:
        print(f"[ERROR] Failed downloading {url}: {e}")
        return False


def sync_missing_specs(
    target_specs: Optional[List[str]] = None,
    specs_dir: str = "data/specs",
    download_format: str = "all",  # 'openapi', 'pdf', or 'all'
) -> str:
    """
    Check existing documents in data/specs and download only the missing ones.
    - target_specs: list of spec keys (e.g. ['MEC011', 'MEC012']) or None for all cataloged specs.
    - download_format: 'openapi' (ETSI Forge YAML/JSON), 'pdf' (Deliverable PDF), or 'all'.
    """
    local_dir = Path(specs_dir).resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    local_catalog = scan_local_specs(local_dir)
    to_check = target_specs if target_specs else list(MEC_SPEC_CATALOG.keys())

    report_lines = [
        f"=== ETSI MEC Document Synchronization ===",
        f"Specs Directory: {local_dir}",
        f"Locally Detected Specs: {len(local_catalog)} unique specification(s)",
        "-" * 55,
    ]

    already_present = []
    downloaded = []
    failed = []

    for spec_key in to_check:
        clean_key = spec_key.upper().strip()
        if not clean_key.startswith("MEC"):
            clean_key = f"MEC{clean_key}"

        meta = MEC_SPEC_CATALOG.get(clean_key, {"name": clean_key, "forge_repo": None})
        existing_files = local_catalog.get(clean_key, [])

        if existing_files:
            rel_names = [p.name for p in existing_files]
            already_present.append(f"[EXISTS] {clean_key} ({meta['name']}): {', '.join(rel_names)}")
            continue

        # Spec is missing! Download missing document
        print(f"[*] Missing {clean_key} ({meta['name']}). Attempting retrieval...", flush=True)
        success = False

        # 1. Download OpenAPI specification from ETSI Forge if repo is available
        forge_repo = meta.get("forge_repo")
        if forge_repo and download_format in ["openapi", "all"]:
            versions = forge_client.get_spec_versions(forge_repo)
            latest_ref = versions[0]["version"] if versions else "master"
            tree = forge_client.list_files(forge_repo, ref=latest_ref)
            api_files = [f for f in tree if f.get("type") == "blob" and (f["name"].endswith(".yaml") or f["name"].endswith(".json"))]

            if api_files:
                out_dir = local_dir / "openapi" / forge_repo / latest_ref
                for item in api_files:
                    dest = out_dir / item["name"]
                    if forge_client.download_raw_file(forge_repo, item["path"], latest_ref, dest):
                        success = True
                if success:
                    downloaded.append(f"[DOWNLOADED FORGE OPENAPI] {clean_key} -> {out_dir.name} ({latest_ref})")

        # 2. If PDF requested or no forge repo, search for official PDF deliverable
        if download_format in ["pdf", "all"] and not success:
            pdf_url = find_etsi_pdf_url(f"GS {clean_key}")
            if pdf_url:
                pdf_filename = pdf_url.split("/")[-1]
                pdf_dest = local_dir / pdf_filename
                if download_direct_pdf(pdf_url, pdf_dest):
                    downloaded.append(f"[DOWNLOADED PDF] {clean_key} -> {pdf_filename}")
                    success = True

        if not success:
            failed.append(f"[UNAVAILABLE] {clean_key} ({meta['name']}) - no direct download found.")

    report_lines.extend(already_present)
    if downloaded:
        report_lines.append("\nNewly Downloaded Missing Specs:")
        report_lines.extend(downloaded)
    if failed:
        report_lines.append("\nCould not automatically download:")
        report_lines.extend(failed)

    report_lines.append("-" * 55)
    report_lines.append(f"Summary: {len(already_present)} existing, {len(downloaded)} newly downloaded, {len(failed)} pending.")
    return "\n".join(report_lines)


# Native Haystack Tool
sync_missing_specs_tool = Tool(
    name="sync_missing_etsi_specs",
    description="Scan the local ETSI data directory (data/specs), detect missing MEC specifications, and download only the missing documents or OpenAPI files from ETSI Forge.",
    parameters={
        "type": "object",
        "properties": {
            "target_specs": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific specs to check (e.g. ['MEC011', 'MEC012']). If omitted, all standard MEC specs are checked.",
            },
            "specs_dir": {
                "type": "string",
                "description": "Directory where specs are stored (default: 'data/specs').",
            },
            "download_format": {
                "type": "string",
                "enum": ["all", "openapi", "pdf"],
                "description": "Format to download: 'all', 'openapi', or 'pdf'.",
            },
        },
    },
    function=sync_missing_specs,
)


def main():
    parser = argparse.ArgumentParser(description="Sync ETSI MEC specifications, downloading only missing ones.")
    parser.add_argument("specs", nargs="*", help="Specific spec identifiers to sync (e.g. MEC012 MEC013)")
    parser.add_argument("--dir", default="data/specs", help="Target specs directory")
    parser.add_argument("--format", choices=["all", "openapi", "pdf"], default="all", help="Format to download")

    args = parser.parse_args()
    report = sync_missing_specs(
        target_specs=args.specs if args.specs else None,
        specs_dir=args.dir,
        download_format=args.format,
    )
    print(report)


if __name__ == "__main__":
    main()
