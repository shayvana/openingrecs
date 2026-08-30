"""
Serialize a NetworkResult to data/network.json with build metadata.

The JSON file is the single artifact read by the Flask app at startup.
A versioned copy (data/network_YYYY-MM.json) is kept alongside it.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

from pipeline.build import NetworkResult

logger = logging.getLogger(__name__)


def export(
    result: NetworkResult,
    month: str,
    data_dir: str = "data",
) -> str:
    """
    Write result to data/network.json (and a versioned backup).

    Returns the path of the primary output file.
    """
    os.makedirs(data_dir, exist_ok=True)

    payload = {
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "month": month,
        "games_processed": result.games_processed,
        "num_players": result.num_players,
        "node_count": result.num_openings,
        "edge_count": len(result.edges),
        "nodes": result.nodes,
        "edges": result.edges,
    }

    primary = os.path.join(data_dir, "network.json")
    versioned = os.path.join(data_dir, f"network_{month}.json")

    for path in (primary, versioned):
        with open(path, "w") as f:
            json.dump(payload, f, separators=(",", ":"))
        size_kb = os.path.getsize(path) / 1024
        logger.info("Wrote %s (%.0f KB)", path, size_kb)

    return primary
