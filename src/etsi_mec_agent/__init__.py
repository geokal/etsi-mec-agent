from etsi_mec_agent.config import settings

__all__ = ["settings"]


def main():
    print("ETSI MEC Haystack Agent")
    print("=" * 40)
    print("Pipelines:")
    print("  Ingest specs:  uv run python -m etsi_mec_agent.ingest [data/specs]")
    print("  Search specs:  uv run python -m etsi_mec_agent.search 'What is Mp1 reference point?'")
