"""
Convert the existing relatedness_network_v2.pkl + metadata to data/network.json.

Run once to bootstrap the new JSON-based app before the full pipeline runs:
    python3 scripts/pkl_to_json.py
"""

import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx

PKL_PATH = "data/relatedness_network_v2.pkl"
META_PATH = "data/relatedness_network_v2_metadata.pkl"
OUT_PATH = "data/network.json"


def main():
    print(f"Loading {PKL_PATH}...")
    with open(PKL_PATH, "rb") as f:
        G = pickle.load(f)

    print(f"Loading {META_PATH}...")
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)

    complexity = meta.get("opening_complexity", {})

    if "?" in G:
        G.remove_node("?")

    nodes = {}
    for name, attrs in G.nodes(data=True):
        nodes[name] = {
            "complexity": float(complexity.get(name, attrs.get("complexity", 1.0))),
            "play_count": int(attrs.get("play_count", 0)),
            "player_diversity": int(attrs.get("player_diversity", 0)),
        }

    edges = []
    for a, b, attrs in G.edges(data=True):
        edges.append({
            "a": a,
            "b": b,
            "weight": float(attrs.get("weight", 1.0)),
            "z_score": float(attrs.get("z_score", 0.0)),
        })

    payload = {
        "built_at": "2024-04-15T00:00:00Z",
        "month": "2024-04",
        "games_processed": meta.get("num_players", 0),  # approximate
        "num_players": meta.get("num_players", 0),
        "node_count": G.number_of_nodes(),
        "edge_count": G.number_of_edges(),
        "nodes": nodes,
        "edges": edges,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, separators=(",", ":"))

    size_kb = os.path.getsize(OUT_PATH) / 1024
    print(f"Wrote {OUT_PATH} ({size_kb:.0f} KB)")
    print(f"  {G.number_of_nodes()} openings, {G.number_of_edges()} edges")


if __name__ == "__main__":
    main()
