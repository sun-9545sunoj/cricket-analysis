"""Download Cricsheet data, rebuild the DuckDB database and retrain the models.

    python -m src.pipeline              # download archives if missing, ingest, train
    python -m src.pipeline --refresh    # re-download the latest archives first (picks up new matches)
    python -m src.pipeline --skip-train
"""
import argparse
import time

from src.config import ARCHIVES
from src.etl import download_archives, ingest
from src.models import train_all


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refresh", action="store_true", help="re-download archives from Cricsheet")
    parser.add_argument("--formats", nargs="+", choices=list(ARCHIVES), default=list(ARCHIVES))
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    start = time.time()
    download_archives(args.formats, refresh=args.refresh)
    count = ingest(formats=args.formats)
    print(f"Ingested {count} matches in {time.time() - start:.1f}s")

    if not args.skip_train:
        train_all(args.formats)
    print(f"Pipeline finished in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
