import logging
import math
import re
from collections import defaultdict
from statistics import median
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Matches trailing Chess.com move-annotation suffixes like "...6.Nc3 Be7" or "2.g3"
_MOVE_SUFFIX_RE = re.compile(r'(\.\.\.|(?:\s+\d+\.\S+))\s*.*$')
# Characters to strip when doing fuzzy matching (apostrophes, curly quotes, hyphens)
_FUZZY_STRIP_RE = re.compile(r"['’\-]")


def normalize_opening(name: str) -> str:
    """Strip variation details from a Lichess opening name for network lookup."""
    if ':' in name:
        name = name.split(':')[0].strip()
    return name.replace('(White)', '').replace('(Black)', '').strip()


class RecommendationEngine:
    """
    Ranks candidate openings for a player based on network proximity,
    rarity, popularity, and novelty.

    The network is a BiCM-filtered bipartite projection over Lichess games.
    Edge weights are co-occurrence counts; z_scores measure statistical
    significance relative to the null model. Node 'complexity' values are
    EFC rarity scores (mean ≈ 1.0) — higher means rarer, not harder.
    """

    def __init__(
        self,
        network,
        rarity_weight: float = 0.3,
        similarity_weight: float = 0.4,
        popularity_weight: float = 0.2,
        novelty_weight: float = 0.1,
    ):
        self.network = network
        raw = {
            'rarity': rarity_weight,
            'similarity': similarity_weight,
            'popularity': popularity_weight,
            'novelty': novelty_weight,
        }
        total = sum(raw.values())
        self.weights = {k: v / total for k, v in raw.items()}

        # Pre-build a fuzzy lookup: stripped-lowercase node name → actual node name.
        # Used by _to_network_node to handle Chess.com naming divergences.
        self._node_lookup: Dict[str, str] = {
            _FUZZY_STRIP_RE.sub('', node.lower()): node
            for node in network.nodes()
        }
        logger.info("RecommendationEngine ready (%d openings)", network.number_of_nodes())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate_rarity(
        self,
        user_openings: List[str],
        user_rating: Optional[int] = None,
    ) -> float:
        """
        Median EFC rarity score across the user's network-matched openings.
        Falls back to a rating-derived estimate when none of their openings
        appear in the network.
        """
        nodes = [self._to_network_node(o) for o in user_openings]
        scores = [
            self.network.nodes[n].get('complexity', 1.0)
            for n in nodes
            if n is not None
        ]
        if scores:
            return float(median(scores))
        if user_rating:
            return (user_rating - 1000) / 1000
        return 1.0

    def recommend(
        self,
        user_openings: List[str],
        user_rating: Optional[int] = None,
        top_k: int = 10,
    ) -> List[Tuple[str, float, str, int]]:
        """
        Return the top-k recommended openings as
        (opening_name, score, explanation, n_supporting).

        n_supporting is the count of the user's openings that connect to
        each candidate in the network.
        """
        user_rarity = self.estimate_rarity(user_openings, user_rating)
        candidates, n_supporting = self._find_candidates(user_openings)

        if not candidates:
            logger.warning("No candidates found for user openings")
            return []

        scores = {
            opening: (
                self.weights['similarity']  * candidates[opening] +
                self.weights['rarity']      * self._score_by_rarity(candidates, user_rarity).get(opening, 0) +
                self.weights['popularity']  * self._score_by_popularity(candidates).get(opening, 0) +
                self.weights['novelty']     * self._score_by_novelty(candidates, user_openings).get(opening, 0)
            )
            for opening in candidates
        }

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            (opening, score, self._explain(opening, user_openings), n_supporting.get(opening, 1))
            for opening, score in ranked
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_network_node(self, opening: str) -> Optional[str]:
        """
        Map any opening name to a network node, tolerating variation detail.

        Handles:
        - Lichess colon-style:   "French Defense: Classical Variation" → "French Defense"
        - Chess.com move suffix: "Italian Game...6.Nc3 Be7 7.O-O" → "Italian Game"
        - Chess.com no colon:    "Sicilian Defense Najdorf Variation" → "Sicilian Defense"
        - Fuzzy apostrophe:      "Queens Gambit Declined" → "Queen's Gambit Declined"
        """
        # Standard normalization (handles Lichess colon-style)
        n = normalize_opening(opening)
        if n in self.network.nodes:
            return n

        # Strip Chess.com move-annotation suffixes ("...6.Nc3 ..." or trailing " 2.g3")
        n = _MOVE_SUFFIX_RE.sub('', n).strip()
        if n in self.network.nodes:
            return n

        # Fuzzy prefix search: strip apostrophes/hyphens, lowercase, try shorter prefixes
        words = _FUZZY_STRIP_RE.sub('', n.lower()).split()
        for length in range(len(words), 0, -1):
            key = ' '.join(words[:length])
            if key in self._node_lookup:
                return self._node_lookup[key]

        return None

    # ------------------------------------------------------------------
    # Internal scoring
    # ------------------------------------------------------------------

    def _find_candidates(
        self,
        user_openings: List[str],
        max_candidates: int = 50,
    ) -> Tuple[Dict[str, float], Dict[str, int]]:
        """
        Walk the network from each of the user's openings and accumulate
        similarity scores for their neighbors.

        Returns (similarity_scores, n_supporting) where n_supporting[c]
        is the count of distinct user openings that connect to candidate c.
        """
        node_for = {o: self._to_network_node(o) for o in user_openings}
        in_network = [node for node in node_for.values() if node is not None]

        if not in_network:
            logger.warning("None of the user's openings matched the network")
            return {}, {}

        similarity: Dict[str, float] = defaultdict(float)
        supporting: Dict[str, set] = defaultdict(set)
        in_network_set = set(in_network)

        for user_opening in in_network:
            for neighbor in self.network.neighbors(user_opening):
                if neighbor in in_network_set or not neighbor or len(neighbor) <= 1:
                    continue
                edge = self.network[user_opening][neighbor]
                z = edge.get('z_score', 1.0)
                weight = edge.get('weight', 1.0)
                similarity[neighbor] += weight * (1 + math.log(max(z, 1.0)))
                supporting[neighbor].add(user_opening)

        if not similarity:
            return {}, {}

        max_sim = max(similarity.values())
        normalized_sim = {k: v / max_sim for k, v in similarity.items()}
        top = dict(
            sorted(normalized_sim.items(), key=lambda x: x[1], reverse=True)[:max_candidates]
        )
        return top, {k: len(supporting[k]) for k in top}

    def _score_by_rarity(
        self,
        candidates: Dict[str, float],
        user_rarity: float,
        tolerance: float = 0.5,
    ) -> Dict[str, float]:
        """
        Score each candidate by how close its EFC rarity is to the user's,
        with a slight upward nudge (×1.2) toward less-familiar openings.

        Note: scores rarity proximity only — not chess difficulty.
        """
        target = user_rarity * 1.2
        scores = {}
        for opening in candidates:
            if opening not in self.network.nodes:
                continue
            r = self.network.nodes[opening].get('complexity', 1.0)
            distance = abs(r - target) / max(target, 0.01)
            if distance <= tolerance:
                scores[opening] = 1.0 - distance / tolerance
            else:
                scores[opening] = math.exp(-(distance - tolerance))
        return scores

    def _score_by_popularity(self, candidates: Dict[str, float]) -> Dict[str, float]:
        """Score each candidate by log-popularity (degree, or play_count × diversity)."""
        raw = {}
        for opening in candidates:
            if opening not in self.network.nodes:
                continue
            node = self.network.nodes[opening]
            play_count = node.get('play_count', 0)
            diversity = node.get('player_diversity', 0)
            if play_count > 0 and diversity > 0:
                raw[opening] = math.log(play_count + 1) * math.log(diversity + 1)
            else:
                raw[opening] = math.log(self.network.degree(opening) + 1)

        if not raw:
            return {}
        max_val = max(raw.values())
        return {k: v / max_val for k, v in raw.items()} if max_val > 0 else raw

    def _score_by_novelty(
        self,
        candidates: Dict[str, float],
        user_openings: List[str],
    ) -> Dict[str, float]:
        """Score each candidate by word-overlap novelty vs. the user's repertoire."""
        user_word_sets = [set(normalize_opening(o).lower().split()) for o in user_openings]
        scores = {}
        for opening in candidates:
            words = set(opening.lower().split())
            max_overlap = max((len(words & u) for u in user_word_sets), default=0)
            scores[opening] = 1.0 / (1.0 + max_overlap)
        return scores

    def _explain(self, opening: str, user_openings: List[str]) -> str:
        """Build a one-sentence explanation for a recommended opening."""
        user_nodes = [self._to_network_node(o) for o in user_openings]
        connections = sorted(
            [
                (u, self.network[u][opening].get('z_score', 1.0))
                for u in user_nodes
                if u and u in self.network and opening in self.network[u]
            ],
            key=lambda x: -x[1],
        )

        parts = []
        if connections:
            name, z = connections[0]
            prefix = "strongly " if z > 100 else ""
            parts.append(f"{prefix}linked to {name} (z={z:.0f})")

        if opening in self.network.nodes:
            node = self.network.nodes[opening]
            rarity = node.get('complexity', 1.0)
            diversity = node.get('player_diversity', 0)
            if rarity < 0.1:
                parts.append("one of the most-played openings in the corpus")
            elif rarity > 5.0:
                parts.append("rare in the corpus — fewer resources and courses available")
            if diversity > 100:
                parts.append(f"seen from {diversity:,} distinct players")

        if not parts:
            return "Matched by network proximity to your current repertoire."
        return "; ".join(parts).capitalize() + "."
