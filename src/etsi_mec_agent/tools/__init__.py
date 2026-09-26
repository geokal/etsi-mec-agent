def __getattr__(name: str):
    if name in (
        "ETSIForgeClient",
        "forge_client",
        "check_etsi_spec_versions",
        "download_forge_openapi",
        "etsi_forge_versions_tool",
        "etsi_forge_download_tool",
    ):
        from etsi_mec_agent.tools import etsi_forge
        return getattr(etsi_forge, name)

    if name in ("search_etsi_web", "find_etsi_pdf_url", "exa_etsi_search_tool"):
        from etsi_mec_agent.tools import exa_search
        return getattr(exa_search, name)

    if name in ("scan_local_specs", "sync_missing_specs", "sync_missing_specs_tool"):
        from etsi_mec_agent.tools import spec_sync
        return getattr(spec_sync, name)

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "ETSIForgeClient",
    "forge_client",
    "check_etsi_spec_versions",
    "download_forge_openapi",
    "etsi_forge_versions_tool",
    "etsi_forge_download_tool",
    "search_etsi_web",
    "find_etsi_pdf_url",
    "exa_etsi_search_tool",
    "scan_local_specs",
    "sync_missing_specs",
    "sync_missing_specs_tool",
]
