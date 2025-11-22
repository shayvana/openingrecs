"""
Similarity Metrics for Chess Openings

Implements similarity calculations as described in the Nature paper.

Similarity methods:
1. Jaccard similarity: Based on common players
2. Cosine similarity: Based on player vectors
3. Co-occurrence based: Statistical co-occurrence in games

References:
    - Nature paper Section: "Similarity Metrics"
"""

import numpy as np
from typing import Set, Dict, List, Tuple
from scipy.spatial.distance import cosine, jaccard
import logging

logger = logging.getLogger(__name__)


class SimilarityCalculator:
    """
    Calculate similarity between chess openings using various metrics.
    """

    def jaccard_similarity(
        self,
        set1: Set,
        set2: Set
    ) -> float:
        """
        Calculate Jaccard similarity between two sets.

        Jaccard similarity = |A ∩ B| / |A ∪ B|

        Args:
            set1: First set (e.g., players who play opening 1)
            set2: Second set (e.g., players who play opening 2)

        Returns:
            Jaccard similarity score [0, 1]
        """
        if not set1 and not set2:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        if union == 0:
            return 0.0

        return intersection / union

    def cosine_similarity(
        self,
        vector1: np.ndarray,
        vector2: np.ndarray
    ) -> float:
        """
        Calculate cosine similarity between two vectors.

        Cosine similarity = (A · B) / (||A|| ||B||)

        Args:
            vector1: First vector (e.g., player usage vector for opening 1)
            vector2: Second vector (e.g., player usage vector for opening 2)

        Returns:
            Cosine similarity score [-1, 1] (typically [0, 1] for non-negative vectors)
        """
        if len(vector1) != len(vector2):
            raise ValueError("Vectors must have same length")

        # Handle zero vectors
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # scipy.spatial.distance.cosine returns distance, not similarity
        # similarity = 1 - distance
        return 1 - cosine(vector1, vector2)

    def calculate_pairwise_jaccard(
        self,
        opening_players: Dict[str, Set[str]]
    ) -> Dict[Tuple[str, str], float]:
        """
        Calculate Jaccard similarity for all pairs of openings.

        Args:
            opening_players: Dictionary mapping opening names to sets of players

        Returns:
            Dictionary mapping (opening1, opening2) tuples to similarity scores
        """
        openings = list(opening_players.keys())
        similarities = {}

        for i, opening1 in enumerate(openings):
            for j, opening2 in enumerate(openings):
                if i < j:  # Only calculate for unique pairs
                    players1 = opening_players[opening1]
                    players2 = opening_players[opening2]
                    sim = self.jaccard_similarity(players1, players2)
                    similarities[(opening1, opening2)] = sim

        logger.info(f"Calculated Jaccard similarity for {len(similarities)} opening pairs")
        return similarities

    def calculate_pairwise_cosine(
        self,
        opening_vectors: Dict[str, np.ndarray]
    ) -> Dict[Tuple[str, str], float]:
        """
        Calculate cosine similarity for all pairs of openings.

        Args:
            opening_vectors: Dictionary mapping opening names to player usage vectors

        Returns:
            Dictionary mapping (opening1, opening2) tuples to similarity scores
        """
        openings = list(opening_vectors.keys())
        similarities = {}

        for i, opening1 in enumerate(openings):
            for j, opening2 in enumerate(openings):
                if i < j:  # Only calculate for unique pairs
                    vec1 = opening_vectors[opening1]
                    vec2 = opening_vectors[opening2]
                    sim = self.cosine_similarity(vec1, vec2)
                    similarities[(opening1, opening2)] = sim

        logger.info(f"Calculated cosine similarity for {len(similarities)} opening pairs")
        return similarities

    def calculate_weighted_similarity(
        self,
        opening1: str,
        opening2: str,
        player_sets: Dict[str, Set[str]],
        player_weights: Dict[str, float]
    ) -> float:
        """
        Calculate weighted Jaccard similarity using player importance weights.

        Players with higher ratings/success contribute more to similarity.

        Args:
            opening1: Name of first opening
            opening2: Name of second opening
            player_sets: Dictionary mapping openings to player sets
            player_weights: Dictionary mapping player names to importance weights

        Returns:
            Weighted similarity score
        """
        if opening1 not in player_sets or opening2 not in player_sets:
            return 0.0

        players1 = player_sets[opening1]
        players2 = player_sets[opening2]

        # Calculate weighted intersection and union
        intersection = players1 & players2
        union = players1 | players2

        if not union:
            return 0.0

        weighted_intersection = sum(player_weights.get(p, 1.0) for p in intersection)
        weighted_union = sum(player_weights.get(p, 1.0) for p in union)

        if weighted_union == 0:
            return 0.0

        return weighted_intersection / weighted_union

    def create_opening_vectors(
        self,
        bipartite_matrix: np.ndarray,
        players: List[str],
        openings: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        Create player usage vectors for each opening.

        Args:
            bipartite_matrix: Matrix of shape (n_players, n_openings)
            players: List of player names (rows)
            openings: List of opening names (columns)

        Returns:
            Dictionary mapping opening names to player usage vectors
        """
        opening_vectors = {}

        for j, opening in enumerate(openings):
            # Extract column for this opening
            vector = bipartite_matrix[:, j]
            opening_vectors[opening] = vector

        return opening_vectors

    def calculate_similarity_matrix(
        self,
        co_occurrence_matrix: np.ndarray,
        method: str = 'jaccard'
    ) -> np.ndarray:
        """
        Calculate similarity matrix from co-occurrence matrix.

        Args:
            co_occurrence_matrix: Square matrix of co-occurrences
            method: Similarity method ('jaccard', 'cosine', 'dice')

        Returns:
            Similarity matrix of same shape
        """
        n = co_occurrence_matrix.shape[0]
        similarity_matrix = np.zeros((n, n))

        # Get diagonal (self co-occurrence counts)
        diagonal = np.diag(co_occurrence_matrix)

        for i in range(n):
            for j in range(i, n):
                if method == 'jaccard':
                    # Jaccard: |A ∩ B| / |A ∪ B|
                    # |A ∪ B| = |A| + |B| - |A ∩ B|
                    intersection = co_occurrence_matrix[i, j]
                    union = diagonal[i] + diagonal[j] - intersection
                    similarity = intersection / union if union > 0 else 0.0

                elif method == 'cosine':
                    # Cosine: (A · B) / (||A|| ||B||)
                    numerator = co_occurrence_matrix[i, j]
                    denominator = np.sqrt(diagonal[i] * diagonal[j])
                    similarity = numerator / denominator if denominator > 0 else 0.0

                elif method == 'dice':
                    # Dice coefficient: 2|A ∩ B| / (|A| + |B|)
                    intersection = co_occurrence_matrix[i, j]
                    sum_sizes = diagonal[i] + diagonal[j]
                    similarity = 2 * intersection / sum_sizes if sum_sizes > 0 else 0.0

                else:
                    raise ValueError(f"Unknown similarity method: {method}")

                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity  # Symmetric

        return similarity_matrix


def calculate_overlap_coefficient(set1: Set, set2: Set) -> float:
    """
    Calculate overlap coefficient (Szymkiewicz–Simpson coefficient).

    Overlap = |A ∩ B| / min(|A|, |B|)

    This measures the degree to which the smaller set is contained in the larger.

    Args:
        set1: First set
        set2: Second set

    Returns:
        Overlap coefficient [0, 1]
    """
    if not set1 or not set2:
        return 0.0

    intersection = len(set1 & set2)
    min_size = min(len(set1), len(set2))

    return intersection / min_size if min_size > 0 else 0.0


def calculate_correlation_similarity(
    vector1: np.ndarray,
    vector2: np.ndarray
) -> float:
    """
    Calculate Pearson correlation coefficient between two vectors.

    Measures linear relationship between usage patterns.

    Args:
        vector1: First vector
        vector2: Second vector

    Returns:
        Correlation coefficient [-1, 1]
    """
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must have same length")

    # Need at least 2 data points for correlation
    if len(vector1) < 2:
        return 0.0

    # Remove pairs where both are zero (no information)
    mask = (vector1 != 0) | (vector2 != 0)
    if not np.any(mask):
        return 0.0

    v1 = vector1[mask]
    v2 = vector2[mask]

    # Calculate correlation
    if len(v1) < 2:
        return 0.0

    correlation = np.corrcoef(v1, v2)[0, 1]

    # Handle NaN (e.g., if one vector is constant)
    if np.isnan(correlation):
        return 0.0

    return correlation


def rank_similar_openings(
    target_opening: str,
    similarity_scores: Dict[Tuple[str, str], float],
    top_k: int = 10
) -> List[Tuple[str, float]]:
    """
    Rank openings by similarity to a target opening.

    Args:
        target_opening: Name of the opening to find similarities for
        similarity_scores: Dictionary of pairwise similarities
        top_k: Number of top similar openings to return

    Returns:
        List of (opening_name, similarity_score) tuples, sorted by similarity
    """
    relevant_scores = []

    for (opening1, opening2), score in similarity_scores.items():
        if opening1 == target_opening:
            relevant_scores.append((opening2, score))
        elif opening2 == target_opening:
            relevant_scores.append((opening1, score))

    # Sort by similarity (descending)
    relevant_scores.sort(key=lambda x: x[1], reverse=True)

    return relevant_scores[:top_k]
