"""
Chess opening network pipeline — download, build, export.

Usage:
    python -m pipeline.run 2025-01 500000
    python -m pipeline.run 2025-01 500000 --min-rating 1800 --z-threshold 2.5
    python -m pipeline.run 2025-01 500000 --data-dir /tmp/data

The pipeline streams the Lichess .pgn.zst dump for the given month,
builds the opening co-occurrence network, runs BiCM filtering and EFC
rarity scoring, and writes data/network.json (+ a versioned copy).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Allow running from the repo root as: python -m pipeline.run
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.build import build_network
from pipeline.export import export
from pipeline.extract import stream_games

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild the chess opening network")
    parser.add_argument("month", help="Month to process, e.g. 2025-01")
    parser.add_argument("max_games", type=int, help="Maximum games to stream")
    parser.add_argument(
        "--min-rating", type=int, default=1500,
        help="Minimum player rating (default: 1500)",
    )
    parser.add_argument(
        "--z-threshold", type=float, default=2.0,
        help="BiCM z-score threshold for edge retention (default: 2.0)",
    )
    parser.add_argument(
        "--data-dir", default="data",
        help="Directory to write network.json (default: data/)",
    )
    args = parser.parse_args()

    logger.info(
        "Pipeline start | month=%s max_games=%d min_rating=%d z_threshold=%.1f",
        args.month, args.max_games, args.min_rating, args.z_threshold,
    )
    t0 = time.monotonic()

    games = stream_games(args.month, args.max_games, args.min_rating)
    result = build_network(games, z_threshold=args.z_threshold)
    output = export(result, month=args.month, data_dir=args.data_dir)

    elapsed = time.monotonic() - t0
    logger.info(
        "Pipeline complete in %.1fs | %d openings | %d edges → %s",
        elapsed, result.num_openings, len(result.edges), output,
    )
    print(f"\nDone. Network written to {output}")
    print(f"  {result.num_openings} openings, {len(result.edges)} edges")
    print(f"  {result.games_processed:,} games processed in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
