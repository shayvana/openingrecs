"""
Lightweight chess opening co-occurrence network, loaded from a JSON artifact.

Replaces NetworkX for the Flask app — provides the same access patterns
the recommendation engine uses, with no heavy external dependencies.

JSON format expected at data/network.json:
  {
    "built_at":        ISO-8601 string,
    "month":           "YYYY-MM",
    "games_processed": int,
    "nodes": {
      "Sicilian Defense": {"complexity": 0.0003, "play_count": 50000, "player_diversity": 15000},
      ...
    },
    "edges": [
      {"a": "Sicilian Defense", "b": "French Defense", "weight": 1234.5, "z_score": 45.2},
      ...
    ]
  }
"""

import json
from typing import Iterator


class _NodeView:
    """Dict-like view of node attributes; supports 'in', [], and iteration."""

    __slots__ = ("_nodes",)

    def __init__(self, nodes: dict) -> None:
        self._nodes = nodes

    # Make network.nodes() (called as a function) work the same as network.nodes
    def __call__(self) -> "_NodeView":
        return self

    def __contains__(self, node: object) -> bool:
        return node in self._nodes

    def __getitem__(self, node: str) -> dict:
        return self._nodes[node]

    def __setitem__(self, node: str, attrs: dict) -> None:
        self._nodes[node] = attrs

    def __iter__(self) -> Iterator[str]:
        return iter(self._nodes)

    def __len__(self) -> int:
        return len(self._nodes)

    def pop(self, node: str, *args) -> dict:
        return self._nodes.pop(node, *args)

    def items(self):
        return self._nodes.items()


class ChessNetwork:
    """
    Opening co-occurrence network loaded from a JSON artifact.

    Implements the subset of the NetworkX Graph API used by the
    recommendation engine and Flask routes:
      - network.nodes                 → NodeView (supports 'in', [name], iteration)
      - network[a][b]['z_score']      → edge attribute
      - network.neighbors(node)       → iterable of neighbor names
      - network.degree(node)          → int
      - network.number_of_nodes()     → int
      - network.number_of_edges()     → int
      - network.remove_node(node)     → modifies in place
      - 'node' in network             → bool
    """

    def __init__(self, node_attrs: dict, edges: list) -> None:
        self.nodes = _NodeView(node_attrs)
        # Adjacency: {node: {neighbor: {"weight": float, "z_score": float}}}
        self._adj: dict[str, dict] = {node: {} for node in node_attrs}
        self._edge_count = 0

        for edge in edges:
            a, b = edge["a"], edge["b"]
            if a not in self._adj or b not in self._adj:
                continue
            attrs = {"weight": float(edge["weight"]), "z_score": float(edge["z_score"])}
            self._adj[a][b] = attrs
            self._adj[b][a] = attrs
            self._edge_count += 1

    # ------------------------------------------------------------------
    # NetworkX-compatible interface
    # ------------------------------------------------------------------

    def __contains__(self, node: object) -> bool:
        return node in self.nodes

    def __getitem__(self, node: str) -> dict:
        """network[a] returns the adjacency dict for a, so network[a][b] gives edge attrs."""
        return self._adj.get(node, {})

    def neighbors(self, node: str) -> Iterator[str]:
        return iter(self._adj.get(node, {}))

    def degree(self, node: str) -> int:
        return len(self._adj.get(node, {}))

    def number_of_nodes(self) -> int:
        return len(self.nodes)

    def number_of_edges(self) -> int:
        return self._edge_count

    def remove_node(self, node: str) -> None:
        if node not in self.nodes:
            return
        for neighbor in list(self._adj.get(node, {})):
            self._adj[neighbor].pop(node, None)
            self._edge_count -= 1
        self._adj.pop(node, None)
        self.nodes.pop(node, None)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_json(cls, path: str) -> "ChessNetwork":
        with open(path) as f:
            data = json.load(f)
        return cls(data["nodes"], data["edges"])

    @property
    def metadata(self) -> dict:
        """Top-level metadata from the JSON artifact (built_at, month, etc.)."""
        return self._meta

    @classmethod
    def from_json_with_meta(cls, path: str) -> tuple["ChessNetwork", dict]:
        with open(path) as f:
            data = json.load(f)
        net = cls(data["nodes"], data["edges"])
        meta = {k: v for k, v in data.items() if k not in ("nodes", "edges")}
        return net, meta
