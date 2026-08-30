"""
Build the opening co-occurrence network from a stream of game records.

Pipeline:
  1. Accumulate (player, opening, color) triples into a bipartite matrix
  2. Run BiCM to get null-model probabilities
  3. Compute vectorized z-scores (variance via p·(1-p) matrix multiply)
  4. Filter edges where z-score > threshold
  5. Run NHEFC (EFC variant) to compute opening rarity scores

All heavy computation stays in this module; callers pass GameRecord iterables.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Generator

import numpy as np
from bicm import BipartiteGraph

from pipeline.extract import GameRecord

logger = logging.getLogger(__name__)

# Strip colon-variations ("French Defense: Classical" → "French Defense")
# and color tags from Lichess opening names.
_COLON_RE = re.compile(r":.*$")
_COLOR_TAG_RE = re.compile(r"\s*\((White|Black)\)\s*$")


def _clean_opening(name: str) -> str:
    name = _COLON_RE.sub("", name)
    name = _COLOR_TAG_RE.sub("", name)
    return name.strip()


@dataclass
class NetworkResult:
    """Output of build_network()."""

    # Opening → node attributes
    nodes: dict[str, dict]  # {opening: {play_count, player_diversity, complexity}}
    # List of edge dicts {a, b, weight, z_score}
    edges: list[dict]
    # Summary stats
    num_players: int
    num_openings: int
    games_processed: int


def build_network(
    games: Generator[GameRecord, None, None],
    z_threshold: float = 2.0,
) -> NetworkResult:
    """
    Build and filter the opening co-occurrence network from an iterable of GameRecords.

    Returns a NetworkResult with nodes, edges, and summary stats.
    """
    logger.info("Phase 1: accumulating co-occurrence data")

    # player → set of openings they played (any color)
    player_openings: dict[str, set] = defaultdict(set)
    # opening → {play_count, players}
    opening_stats: dict[str, dict] = defaultdict(lambda: {"play_count": 0, "players": set()})
    # (player, opening) → count
    edge_counts: dict[tuple, int] = defaultdict(int)

    games_processed = 0

    for rec in games:
        opening = _clean_opening(rec.opening)
        if not opening:
            continue
        games_processed += 1

        for player in (rec.white, rec.black):
            player_openings[player].add(opening)
            opening_stats[opening]["play_count"] += 1
            opening_stats[opening]["players"].add(player)
            edge_counts[(player, opening)] += 1

    logger.info(
        "Accumulated %d games | %d players | %d openings",
        games_processed,
        len(player_openings),
        len(opening_stats),
    )

    if not player_openings:
        raise ValueError("No games yielded — check the data source and filters")

    # Build ordered lists for matrix indexing
    players = sorted(player_openings)
    openings = sorted(opening_stats)
    player_idx = {p: i for i, p in enumerate(players)}
    opening_idx = {o: i for i, o in enumerate(openings)}
    n_players = len(players)
    n_openings = len(openings)

    logger.info("Phase 2: building bipartite matrix (%d × %d)", n_players, n_openings)
    M = np.zeros((n_players, n_openings), dtype=np.float32)
    for (player, opening), count in edge_counts.items():
        if player in player_idx and opening in opening_idx:
            M[player_idx[player], opening_idx[opening]] = count

    # Co-occurrence: C = Mᵀ @ M (opening × opening), zero diagonal
    logger.info("Phase 3: computing co-occurrence matrix")
    C = M.T @ M
    np.fill_diagonal(C, 0)

    # ------------------------------------------------------------------
    # BiCM null model
    # ------------------------------------------------------------------
    logger.info("Phase 4: fitting BiCM null model")
    binary_M = (M > 0).astype(np.uint8)
    bicm = BipartiteGraph(biadjacency=binary_M)
    bicm.solve_tool()
    P = bicm.get_bicm_matrix().astype(np.float64)  # (n_players, n_openings)

    # Expected co-occurrence: E = Pᵀ @ P
    logger.info("Phase 5: computing expected co-occurrences")
    E = P.T @ P
    np.fill_diagonal(E, 0)

    # Variance (vectorized): Var[C_ij] = Σ_k p_ki·p_kj·(1-p_ki)·(1-p_kj)
    #   = (P·(1-P))ᵀ @ (P·(1-P))
    PQ = P * (1.0 - P)
    Var = PQ.T @ PQ
    np.fill_diagonal(Var, 0)
    Std = np.sqrt(Var)

    # Z-scores
    logger.info("Phase 6: computing z-scores and filtering (threshold=%.1f)", z_threshold)
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = np.where(Std > 0, (C - E) / Std, 0.0)
    Z[~np.isfinite(Z)] = 0.0

    # Retain edges where z > threshold AND observed > expected
    keep = (Z > z_threshold) & (C > E)
    retained = int(np.sum(np.triu(keep, k=1)))
    total = int(np.sum(np.triu(C > 0, k=1)))
    logger.info("Retained %d / %d edges (%.1f%%)", retained, total, 100 * retained / max(total, 1))

    # ------------------------------------------------------------------
    # EFC rarity (NHEFC variant from the Nature paper)
    # ------------------------------------------------------------------
    logger.info("Phase 7: computing EFC rarity scores")
    rarity = _nhefc(M, delta=1e-3)  # returns array indexed by opening_idx

    # ------------------------------------------------------------------
    # Assemble output
    # ------------------------------------------------------------------
    nodes: dict[str, dict] = {}
    for opening in openings:
        idx = opening_idx[opening]
        stats = opening_stats[opening]
        nodes[opening] = {
            "complexity": float(rarity[idx]),
            "play_count": int(stats["play_count"]),
            "player_diversity": int(len(stats["players"])),
        }

    edges: list[dict] = []
    for i in range(n_openings):
        for j in range(i + 1, n_openings):
            if keep[i, j]:
                edges.append({
                    "a": openings[i],
                    "b": openings[j],
                    "weight": float(C[i, j]),
                    "z_score": float(Z[i, j]),
                })

    return NetworkResult(
        nodes=nodes,
        edges=edges,
        num_players=n_players,
        num_openings=n_openings,
        games_processed=games_processed,
    )


def _nhefc(
    M: np.ndarray,
    delta: float = 1e-3,
    tolerance: float = 1e-6,
    max_iter: int = 200,
) -> np.ndarray:
    """
    Non-Homogeneous EFC (NHEFC) — opening rarity scores, normalized to mean=1.

    M is (n_players, n_openings), weighted by game counts.
    Returns a (n_openings,) array of rarity values; higher = rarer.
    """
    n_players, n_openings = M.shape

    # Normalize rows so each player's games sum to 1
    row_sums = M.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    N = M / row_sums  # normalized frequency matrix

    P = np.ones(n_openings)
    F = np.ones(n_players)

    for _ in range(max_iter):
        P_prev = P
        F_prev = F
        P = 1.0 + (N / np.maximum(F[:, None], 1e-10)).sum(axis=0)
        F = delta**2 + (N / np.maximum(P[None, :], 1e-10)).sum(axis=1)

        if (np.max(np.abs(P - P_prev) / (P_prev + 1e-10)) < tolerance and
                np.max(np.abs(F - F_prev) / (F_prev + 1e-10)) < tolerance):
            break

    Q = 1.0 / np.maximum(P - 1.0, 1e-10)
    mean_Q = Q.mean()
    return Q / mean_Q if mean_Q > 0 else Q
