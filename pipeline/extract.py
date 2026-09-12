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
import time
from dataclasses import dataclass
from typing import Generator, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.exceptions import ProtocolError as Urllib3ProtocolError
from urllib3.util.retry import Retry
import zstandard as zstd

logger = logging.getLogger(__name__)

LICHESS_BASE = "https://database.lichess.org/standard"
CHUNK_SIZE = 1 << 17  # 128 KB
MAX_ATTEMPTS = 5
_HEADERS = {
    "User-Agent": "chess-opening-recs/1.0 (pipeline; contact: openings@serialexperiment.ing)",
}

# Exceptions that warrant a retry — ProtocolError comes from urllib3 directly
# when zstd reads resp.raw, bypassing requests' exception wrapping layer.
_STREAM_ERRORS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.Timeout,
    Urllib3ProtocolError,
    ConnectionResetError,
)


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(_HEADERS)
    retry = Retry(
        total=MAX_ATTEMPTS,
        backoff_factor=2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods={"GET"},
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s

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


def _iter_games_from_response(
    resp: requests.Response,
    max_games: int,
    min_rating: int,
    skip: int,
) -> Generator[tuple[GameRecord, int], None, None]:
    """
    Parse one HTTP response stream.  Yields (GameRecord, total_seen_count).
    Caller supplies *skip* — games to discard before yielding (resume after retry).
    *total_seen_count* is the running count of games passing filters, used for skip logic.
    """
    dctx = zstd.ZstdDecompressor()
    reader = dctx.stream_reader(resp.raw)
    buf = ""
    seen = 0      # games passing the rating filter so far in this stream (from byte 0)
    skipped_rating = 0

    try:
        while seen < skip + max_games:
            chunk = reader.read(CHUNK_SIZE)
            if not chunk:
                break
            buf += chunk.decode("utf-8", errors="ignore")
            parts = buf.split("\n\n")
            i = 0
            while i + 1 < len(parts) and seen < skip + max_games:
                headers_block = parts[i]
                moves_block = parts[i + 1]
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
                        seen += 1
                        if seen > skip:
                            yield GameRecord(
                                white=white,
                                black=black,
                                opening=opening,
                                white_elo=white_elo,
                                black_elo=black_elo,
                            ), seen
                    i += 2
                else:
                    break
            buf = "\n\n".join(parts[i:])
    finally:
        reader.close()


def stream_games(
    year_month: str,
    max_games: int,
    min_rating: int = 0,
) -> Generator[GameRecord, None, None]:
    """
    Stream up to *max_games* GameRecords from the Lichess monthly dump for *year_month*.

    Decompresses the .zst on-the-fly; never writes the full PGN to disk.
    Games with at least one player below *min_rating* are skipped.
    Retries up to MAX_ATTEMPTS times with exponential backoff on connection errors.
    """
    url = f"{LICHESS_BASE}/lichess_db_standard_rated_{year_month}.pgn.zst"
    logger.info("Streaming from %s (max=%d, min_rating=%d)", url, max_games, min_rating)
    sess = _session()

    yielded = 0
    attempt = 0

    while attempt < MAX_ATTEMPTS:
        attempt += 1
        skip = yielded  # resume past already-yielded games on retry
        if attempt > 1:
            delay = 2 ** (attempt - 1)  # 2, 4, 8, 16 …
            logger.warning(
                "Retry %d/%d after connection error (skip=%d, backoff=%ds)",
                attempt, MAX_ATTEMPTS, skip, delay,
            )
            time.sleep(delay)

        try:
            resp = sess.get(url, stream=True, timeout=(15, 60))
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.error("Attempt %d: download failed: %s", attempt, exc)
            if attempt >= MAX_ATTEMPTS:
                raise
            continue

        try:
            for rec, _ in _iter_games_from_response(resp, max_games, min_rating, skip):
                yield rec
                yielded += 1
                if yielded >= max_games:
                    logger.info("Reached max_games=%d", max_games)
                    return
            # Clean EOF — done
            break
        except _STREAM_ERRORS as exc:
            logger.warning("Attempt %d: stream interrupted at game %d: %s", attempt, yielded, exc)
            if attempt >= MAX_ATTEMPTS:
                raise
            # loop continues; skip=yielded will fast-forward past already-delivered games

    logger.info("Extraction complete: %d games yielded", yielded)


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
