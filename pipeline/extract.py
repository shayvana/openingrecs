"""
Stream games from a Lichess monthly database dump.

Downloads the .pgn.zst file and yields parsed game records without
writing the full decompressed PGN to disk.

Usage (standalone):
    python -m pipeline.extract 2025-01 500000 --min-rating 1500
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass
from typing import Generator, Optional

import requests
import zstandard as zstd

logger = logging.getLogger(__name__)

LICHESS_BASE = "https://database.lichess.org/standard"
CHUNK_SIZE = 1 << 17  # 128 KB

# PGN header tags we care about
_TAG_RE = re.compile(r'\[(\w+)\s+"([^"]*)"\]')
# Move text / game termination — marks end of a game block
_RESULT_RE = re.compile(r'\b(1-0|0-1|1/2-1/2|\*)\b')


@dataclass
class GameRecord:
    white: str
    black: str
    opening: str
    white_elo: Optional[int]
    black_elo: Optional[int]


def _parse_headers(block: str) -> dict[str, str]:
    return {m.group(1): m.group(2) for m in _TAG_RE.finditer(block)}


def _safe_elo(val: str) -> Optional[int]:
    try:
        elo = int(val)
        return elo if 0 < elo < 4000 else None
    except (ValueError, TypeError):
        return None


def stream_games(
    year_month: str,
    max_games: int,
    min_rating: int = 0,
) -> Generator[GameRecord, None, None]:
    """
    Stream up to *max_games* GameRecords from the Lichess monthly dump for *year_month*.

    Decompresses the .zst on-the-fly; never writes the full PGN to disk.
    Games with at least one player below *min_rating* are skipped.
    """
    url = f"{LICHESS_BASE}/lichess_db_standard_rated_{year_month}.pgn.zst"
    logger.info("Streaming from %s (max=%d, min_rating=%d)", url, max_games, min_rating)

    try:
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.error("Download failed: %s", exc)
        raise

    dctx = zstd.ZstdDecompressor()
    reader = dctx.stream_reader(resp.raw)

    buf = ""
    yielded = 0
    skipped_rating = 0

    try:
        while yielded < max_games:
            chunk = reader.read(CHUNK_SIZE)
            if not chunk:
                break
            buf += chunk.decode("utf-8", errors="ignore")

            # Games are separated by blank lines; split on double-newline
            # A PGN game = headers + blank line + moves + blank line
            parts = buf.split("\n\n")

            # Process all complete games; each game spans two parts (headers + moves).
            # We need pairs of parts, keeping the last incomplete one.
            i = 0
            while i + 1 < len(parts) and yielded < max_games:
                headers_block = parts[i]
                moves_block = parts[i + 1]

                # Confirm the moves block ends a game (has a result token)
                if _RESULT_RE.search(moves_block):
                    tags = _parse_headers(headers_block)
                    opening = tags.get("Opening", "").strip()
                    white = tags.get("White", "").strip()
                    black = tags.get("Black", "").strip()

                    if opening and white and black:
                        white_elo = _safe_elo(tags.get("WhiteElo", ""))
                        black_elo = _safe_elo(tags.get("BlackElo", ""))

                        if min_rating > 0:
                            if (white_elo is None or white_elo < min_rating or
                                    black_elo is None or black_elo < min_rating):
                                skipped_rating += 1
                                i += 2
                                continue

                        yield GameRecord(
                            white=white,
                            black=black,
                            opening=opening,
                            white_elo=white_elo,
                            black_elo=black_elo,
                        )
                        yielded += 1

                    i += 2
                else:
                    # Incomplete game at boundary; keep from here
                    break

            buf = "\n\n".join(parts[i:])

    finally:
        reader.close()

    logger.info(
        "Extraction complete: %d games yielded, %d skipped (rating filter)",
        yielded,
        skipped_rating,
    )


def download_and_save(
    year_month: str,
    output_path: str,
    max_games: int,
    min_rating: int = 0,
) -> int:
    """
    Stream games and write them as tab-separated records to *output_path*.

    Format per line: white\tblack\topening\twhite_elo\tblack_elo

    Returns the number of games written.
    """
    count = 0
    with open(output_path, "w") as f:
        f.write("white\tblack\topening\twhite_elo\tblack_elo\n")
        for rec in stream_games(year_month, max_games, min_rating):
            f.write(
                f"{rec.white}\t{rec.black}\t{rec.opening}\t"
                f"{rec.white_elo or ''}\t{rec.black_elo or ''}\n"
            )
            count += 1
    logger.info("Wrote %d records to %s", count, output_path)
    return count


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("month", help="YYYY-MM")
    p.add_argument("max_games", type=int)
    p.add_argument("--min-rating", type=int, default=1500)
    p.add_argument("--output", default="data/games.tsv")
    args = p.parse_args()
    n = download_and_save(args.month, args.output, args.max_games, args.min_rating)
    print(f"Wrote {n} games to {args.output}")
