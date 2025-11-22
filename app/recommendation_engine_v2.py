"""
Opening Recommendation Engine (Corrected Methodology)

Recommends chess openings based on:
1. User's current opening repertoire
2. Opening complexity (from EFC algorithm)
3. Opening similarity (from filtered relatedness network)
4. User's skill level (rating)

Key improvements over v1:
- Uses complexity metrics to recommend appropriate openings
- Considers user's skill level
- Multi-factor scoring system
- Detailed explanations
"""

import networkx as nx
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


def normalize_opening(opening: str) -> str:
    """
    Normalize opening name for matching.

    Args:
        opening: Opening name from user games

    Returns:
        Normalized opening name
    """
    # Remove variation details
    if ':' in opening:
        opening = opening.split(':')[0].strip()

    # Remove color specifications
    opening = opening.replace('(White)', '').replace('(Black)', '').strip()

    return opening


class RecommendationEngine:
    """
    Generate opening recommendations based on complexity and similarity.
    """

    def __init__(
        self,
        relatedness_network: nx.Graph,
        complexity_weight: float = 0.3,
        similarity_weight: float = 0.4,
        popularity_weight: float = 0.2,
        novelty_weight: float = 0.1
    ):
        """
        Initialize recommendation engine.

        Args:
            relatedness_network: NetworkX graph with opening relationships
            complexity_weight: Weight for complexity matching
            similarity_weight: Weight for similarity to user openings
            popularity_weight: Weight for opening popularity
            novelty_weight: Weight for introducing new openings
        """
        self.network = relatedness_network
        self.weights = {
            'complexity': complexity_weight,
            'similarity': similarity_weight,
            'popularity': popularity_weight,
            'novelty': novelty_weight
        }

        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}

        logger.info(f"Initialized recommendation engine with {self.network.number_of_nodes()} openings")

    def estimate_user_complexity(
        self,
        user_openings: List[str],
        user_rating: Optional[int] = None
    ) -> float:
        """
        Estimate user's complexity level from their current openings.

        Args:
            user_openings: List of openings the user plays
            user_rating: Optional user rating

        Returns:
            Estimated complexity level
        """
        # Normalize openings
        normalized = [normalize_opening(o) for o in user_openings]

        # Get complexity scores for user's openings
        complexities = []
        for opening in normalized:
            if opening in self.network.nodes:
                complexity = self.network.nodes[opening].get('complexity', 1.0)
                complexities.append(complexity)

        if not complexities:
            # Default complexity based on rating if available
            if user_rating:
                # Rough mapping: 1500 -> 1.0, 2000 -> 1.5, 2500 -> 2.0
                return (user_rating - 1000) / 1000
            return 1.0  # Default middle complexity

        # Use median complexity of current openings
        user_complexity = np.median(complexities)

        logger.debug(f"Estimated user complexity: {user_complexity:.3f}")

        return user_complexity

    def find_similar_openings(
        self,
        user_openings: List[str],
        max_candidates: int = 50
    ) -> Dict[str, float]:
        """
        Find openings similar to user's current repertoire.

        Uses the relatedness network to find connected openings.

        Args:
            user_openings: List of openings the user plays
            max_candidates: Maximum number of candidates to return

        Returns:
            Dictionary mapping opening names to similarity scores
        """
        normalized = [normalize_opening(o) for o in user_openings]

        # Filter to openings that exist in network
        existing = [o for o in normalized if o in self.network.nodes]

        if not existing:
            logger.warning("None of user's openings found in network")
            return {}

        # Accumulate similarity scores
        similarity_scores = defaultdict(float)

        for user_opening in existing:
            # Get neighbors in relatedness network
            if user_opening not in self.network:
                continue

            for neighbor in self.network.neighbors(user_opening):
                if neighbor not in normalized:  # Don't recommend what they already play
                    edge_data = self.network[user_opening][neighbor]
                    weight = edge_data.get('weight', 1.0)
                    z_score = edge_data.get('z_score', 1.0)

                    # Similarity based on edge weight and statistical significance
                    similarity = weight * (1 + np.log(max(z_score, 1.0)))
                    similarity_scores[neighbor] += similarity

        # Normalize scores
        if similarity_scores:
            max_score = max(similarity_scores.values())
            if max_score > 0:
                similarity_scores = {
                    k: v/max_score
                    for k, v in similarity_scores.items()
                }

        # Sort and limit
        top_similar = dict(
            sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)[:max_candidates]
        )

        logger.debug(f"Found {len(top_similar)} similar openings")

        return top_similar

    def filter_by_complexity(
        self,
        candidates: Dict[str, float],
        target_complexity: float,
        tolerance: float = 0.3
    ) -> Dict[str, float]:
        """
        Filter candidates by complexity appropriateness.

        Recommends openings slightly above user's current level (growth zone).

        Args:
            candidates: Dictionary of candidate openings and scores
            target_complexity: Target complexity level
            tolerance: Acceptable complexity range (±tolerance)

        Returns:
            Dictionary with complexity scores [0, 1]
        """
        complexity_scores = {}

        # Prefer openings slightly above current level
        # Optimal complexity: target + 0.1 (stretch goal)
        optimal_complexity = target_complexity + 0.1

        for opening, _ in candidates.items():
            if opening not in self.network.nodes:
                continue

            opening_complexity = self.network.nodes[opening].get('complexity', 1.0)

            # Calculate how far from optimal
            distance = abs(opening_complexity - optimal_complexity)

            # Score decreases with distance
            if distance <= tolerance:
                score = 1.0 - (distance / tolerance)
            else:
                # Beyond tolerance: exponential decay
                score = np.exp(-(distance - tolerance))

            complexity_scores[opening] = score

        return complexity_scores

    def calculate_popularity_scores(
        self,
        candidates: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate popularity scores for candidates.

        More popular openings might be better for beginners,
        less popular for advanced players seeking novelty.

        Args:
            candidates: Dictionary of candidate openings

        Returns:
            Dictionary with popularity scores [0, 1]
        """
        popularity_scores = {}

        for opening in candidates.keys():
            if opening not in self.network.nodes:
                continue

            play_count = self.network.nodes[opening].get('play_count', 0)
            player_diversity = self.network.nodes[opening].get('player_diversity', 0)

            # Popularity based on both play count and player diversity
            popularity = np.log(play_count + 1) * np.log(player_diversity + 1)
            popularity_scores[opening] = popularity

        # Normalize
        if popularity_scores:
            max_pop = max(popularity_scores.values())
            if max_pop > 0:
                popularity_scores = {k: v/max_pop for k, v in popularity_scores.items()}

        return popularity_scores

    def calculate_novelty_scores(
        self,
        candidates: Dict[str, float],
        user_openings: List[str]
    ) -> Dict[str, float]:
        """
        Calculate novelty scores - how different from user's current repertoire.

        Args:
            candidates: Dictionary of candidate openings
            user_openings: User's current openings

        Returns:
            Dictionary with novelty scores [0, 1]
        """
        normalized_user = set(normalize_opening(o) for o in user_openings)

        novelty_scores = {}

        for opening in candidates.keys():
            # Check if opening family is new
            # (e.g., recommending Sicilian to someone who only plays e4)

            # Simple heuristic: count common words
            opening_words = set(opening.lower().split())

            max_overlap = 0
            for user_opening in normalized_user:
                user_words = set(user_opening.lower().split())
                overlap = len(opening_words & user_words)
                max_overlap = max(max_overlap, overlap)

            # Novelty is inverse of overlap
            # Full overlap (same opening family) = low novelty
            # No overlap = high novelty
            novelty = 1.0 / (1.0 + max_overlap)
            novelty_scores[opening] = novelty

        return novelty_scores

    def recommend(
        self,
        user_openings: List[str],
        user_rating: Optional[int] = None,
        top_k: int = 10,
        include_explanations: bool = True
    ) -> List[Tuple[str, float, Optional[str]]]:
        """
        Generate opening recommendations.

        Args:
            user_openings: List of openings the user currently plays
            user_rating: Optional user rating for personalization
            top_k: Number of recommendations to return
            include_explanations: Whether to generate explanations

        Returns:
            List of (opening_name, score, explanation) tuples
        """
        logger.info(f"Generating recommendations for {len(user_openings)} user openings")

        # Step 1: Estimate user complexity
        user_complexity = self.estimate_user_complexity(user_openings, user_rating)
        logger.debug(f"User complexity: {user_complexity:.3f}")

        # Step 2: Find similar openings
        candidates = self.find_similar_openings(user_openings, max_candidates=50)

        if not candidates:
            logger.warning("No candidates found")
            return []

        logger.debug(f"Found {len(candidates)} candidates")

        # Step 3: Calculate component scores
        similarity_scores = candidates  # Already calculated
        complexity_scores = self.filter_by_complexity(candidates, user_complexity)
        popularity_scores = self.calculate_popularity_scores(candidates)
        novelty_scores = self.calculate_novelty_scores(candidates, user_openings)

        # Step 4: Aggregate scores
        final_scores = {}

        for opening in candidates.keys():
            score = (
                self.weights['similarity'] * similarity_scores.get(opening, 0) +
                self.weights['complexity'] * complexity_scores.get(opening, 0) +
                self.weights['popularity'] * popularity_scores.get(opening, 0) +
                self.weights['novelty'] * novelty_scores.get(opening, 0)
            )
            final_scores[opening] = score

        # Step 5: Rank and select top-k
        ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Step 6: Generate explanations
        results = []
        for opening, score in ranked:
            if include_explanations:
                explanation = self._generate_explanation(
                    opening,
                    user_openings,
                    similarity_scores.get(opening, 0),
                    complexity_scores.get(opening, 0),
                    popularity_scores.get(opening, 0)
                )
            else:
                explanation = None

            results.append((opening, score, explanation))

        logger.info(f"Generated {len(results)} recommendations")

        return results

    def _generate_explanation(
        self,
        opening: str,
        user_openings: List[str],
        similarity: float,
        complexity: float,
        popularity: float
    ) -> str:
        """
        Generate human-readable explanation for recommendation.

        Args:
            opening: Recommended opening
            user_openings: User's current openings
            similarity: Similarity score
            complexity: Complexity score
            popularity: Popularity score

        Returns:
            Explanation string
        """
        # Find which user openings this is related to
        normalized_user = [normalize_opening(o) for o in user_openings]
        related_to = []

        for user_opening in normalized_user:
            if user_opening in self.network and opening in self.network[user_opening]:
                related_to.append(user_opening)

        parts = []

        # Similarity explanation
        if related_to:
            parts.append(f"similar to {', '.join(related_to[:2])}")

        # Complexity explanation
        if opening in self.network.nodes:
            opening_complexity = self.network.nodes[opening].get('complexity', 1.0)
            if complexity > 0.7:
                parts.append(f"appropriate complexity level ({opening_complexity:.2f})")

        # Popularity explanation
        if opening in self.network.nodes:
            player_diversity = self.network.nodes[opening].get('player_diversity', 0)
            if player_diversity > 100:
                parts.append(f"played by {player_diversity} players")

        if parts:
            return f"Recommended because it is {' and '.join(parts)}"
        else:
            return "Recommended based on your playing style"


def recommend_openings(
    user_openings: List[str],
    relatedness_network: nx.Graph,
    user_rating: Optional[int] = None,
    top_k: int = 10
) -> Tuple[List[Tuple[str, float]], str]:
    """
    Convenience function for backward compatibility.

    Args:
        user_openings: User's current openings
        relatedness_network: NetworkX graph
        user_rating: Optional user rating
        top_k: Number of recommendations

    Returns:
        Tuple of (recommendations, top_explanation)
    """
    engine = RecommendationEngine(relatedness_network)
    recommendations = engine.recommend(user_openings, user_rating, top_k, include_explanations=True)

    # Format for compatibility
    formatted = [(name, score) for name, score, _ in recommendations]

    # Top explanation
    if recommendations:
        top_explanation = recommendations[0][2]
    else:
        top_explanation = "No recommendations available"

    return formatted, top_explanation
