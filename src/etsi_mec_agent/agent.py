import argparse
import sys
from pathlib import Path

# Import the Haystack tools we created (now located in the repo)
try:
    from .tools.monitor_deliver import monitor_etsi_deliver
    from .tools.etsi_forge import download_specifications_markdown_tool
except Exception as e:
    print(f"Failed to import tools: {e}")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="ETSI MEC agent CLI")
    parser.add_argument("--watch-deliver", type=int, metavar="SECONDS",
                        help="Start the ETSI deliver monitor with the given poll interval (seconds). This call blocks until interrupted.")
    parser.add_argument("--download-markdown", action="store_true",
                        help="Download all markdown specifications from ETSI Forge into the local data directory.")
    args = parser.parse_args()

    if args.watch_deliver:
        print(f"Starting ETSI deliver monitor with interval {args.watch_deliver}s (press Ctrl-C to stop)...")
        try:
            monitor_etsi_deliver(poll_interval=args.watch_deliver)
        except KeyboardInterrupt:
            print("Monitor stopped by user.")
        return

    if args.download_markdown:
        print("Downloading markdown specifications from ETSI Forge...")
        result = download_specifications_markdown_tool()
        print(result)
        return

    parser.print_help()

if __name__ == "__main__":
    main()
