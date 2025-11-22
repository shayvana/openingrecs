"""
Economic Fitness and Complexity (EFC) Algorithm

Implements the methodology from:
"Quantifying the Complexity and Similarity of Chess Openings Using
Online Chess Community Data" (Nature, 2023)

The EFC algorithm measures:
- Player Fitness (F): Capability based on diversity of openings played
- Opening Complexity (Q): Sophistication based on diversity of players who use it

**IMPORTANT NOTE FOR CHESS APPLICATIONS**:
The EFC algorithm produces "rarity" scores, not chess-complexity scores.
- Rare openings (few players) → HIGH EFC complexity
- Popular openings (many players) → LOW EFC complexity

This is backwards for chess intuition:
- Sicilian Defense (99,975 players) → EFC ≈ 0 (but actually very complex!)
- Obscure openings (2 players) → EFC ≈ 144 (but actually simple!)

For chess-specific complexity, consider using:
- Network degree centrality after z-score filtering
- Popularity-weighted metrics
- Theoretical depth (if available)

References:
    - Tacchella, A., et al. (2012). A new metrics for countries' fitness and products' complexity.
    - Original paper Section: "Economic Fitness and Complexity"
"""

import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class EFCCalculator:
    """
    Calculate Economic Fitness and Complexity for chess openings.

    The algorithm iteratively computes:
    1. F_i^(n+1) = Σ_j M_ij * Q_j^(n)  (player fitness)
    2. Q_j^(n+1) = 1 / Σ_i (M_ij / F_i^(n+1))  (opening complexity)

    Where M is the bipartite adjacency matrix (players × openings).
    """

    def __init__(
        self,
        tolerance: float = 1e-6,
        max_iterations: int = 100,
        min_fitness: float = 1e-10
    ):
        """
        Initialize EFC calculator.

        Args:
            tolerance: Convergence threshold for relative change
            max_iterations: Maximum number of iterations
            min_fitness: Minimum fitness value to avoid division by zero
        """
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.min_fitness = min_fitness
        self.converged = False
        self.iterations = 0

    def calculate(
        self,
        bipartite_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate fitness and complexity from bipartite matrix.

        Args:
            bipartite_matrix: Binary or weighted matrix of shape (n_players, n_openings)
                             M[i,j] = 1 if player i played opening j, or weight

        Returns:
            Tuple of (player_fitness, opening_complexity)
            - player_fitness: Array of shape (n_players,)
            - opening_complexity: Array of shape (n_openings,)

        Raises:
            ValueError: If matrix is invalid or algorithm doesn't converge
        """
        # Validate input
        if bipartite_matrix.shape[0] == 0 or bipartite_matrix.shape[1] == 0:
            raise ValueError("Bipartite matrix cannot be empty")

        if not np.any(bipartite_matrix):
            raise ValueError("Bipartite matrix must have at least one non-zero entry")

        n_players, n_openings = bipartite_matrix.shape
        logger.info(f"Starting EFC calculation for {n_players} players and {n_openings} openings")

        # Initialize fitness and complexity uniformly
        # All players start with equal fitness, all openings with equal complexity
        F = np.ones(n_players)
        Q = np.ones(n_openings)

        self.converged = False
        self.iterations = 0

        for iteration in range(self.max_iterations):
            self.iterations = iteration + 1

            # Store previous values for convergence check
            F_prev = F.copy()
            Q_prev = Q.copy()

            # Step 1: Update player fitness
            # F_i = Σ_j M_ij * Q_j
            F_new = bipartite_matrix @ Q

            # Normalize to prevent overflow and maintain scale
            if np.sum(F_new) > 0:
                F_new = F_new / np.mean(F_new)

            # Ensure minimum fitness to avoid division by zero
            F_new = np.maximum(F_new, self.min_fitness)

            # Step 2: Update opening complexity
            # Q_j = 1 / Σ_i (M_ij / F_i)
            # Avoid division by zero by using masked operations
            with np.errstate(divide='ignore', invalid='ignore'):
                # For each opening, sum (M_ij / F_i) over all players
                weighted_sum = np.zeros(n_openings)
                for j in range(n_openings):
                    # Get players who played this opening
                    players_mask = bipartite_matrix[:, j] > 0
                    if np.any(players_mask):
                        weighted_sum[j] = np.sum(
                            bipartite_matrix[players_mask, j] / F_new[players_mask]
                        )

                # Q_j = 1 / weighted_sum
                Q_new = np.zeros(n_openings)
                valid_mask = weighted_sum > 0
                Q_new[valid_mask] = 1.0 / weighted_sum[valid_mask]

            # Normalize complexity
            if np.sum(Q_new) > 0:
                Q_new = Q_new / np.mean(Q_new)

            # Check convergence
            F_change = np.max(np.abs(F_new - F_prev) / (F_prev + 1e-10))
            Q_change = np.max(np.abs(Q_new - Q_prev) / (Q_prev + 1e-10))
            max_change = max(F_change, Q_change)

            logger.debug(f"Iteration {iteration + 1}: max_change = {max_change:.8f}")

            # Update values
            F = F_new
            Q = Q_new

            # Check if converged
            if max_change < self.tolerance:
                self.converged = True
                logger.info(f"EFC converged after {iteration + 1} iterations")
                break

        if not self.converged:
            logger.warning(
                f"EFC did not converge after {self.max_iterations} iterations. "
                f"Final change: {max_change:.8f}"
            )
            raise ValueError(
                f"EFC algorithm did not converge within {self.max_iterations} iterations"
            )

        return F, Q

    def calculate_from_edgelist(
        self,
        player_opening_pairs: list,
        weights: Optional[np.ndarray] = None
    ) -> Tuple[dict, dict]:
        """
        Calculate fitness and complexity from edge list format.

        Args:
            player_opening_pairs: List of (player, opening) tuples
            weights: Optional array of weights for each edge

        Returns:
            Tuple of (player_fitness_dict, opening_complexity_dict)
        """
        # Get unique players and openings
        players = sorted(set(p for p, _ in player_opening_pairs))
        openings = sorted(set(o for _, o in player_opening_pairs))

        # Create index mappings
        player_to_idx = {p: i for i, p in enumerate(players)}
        opening_to_idx = {o: i for i, o in enumerate(openings)}

        # Build bipartite matrix
        n_players = len(players)
        n_openings = len(openings)
        M = np.zeros((n_players, n_openings))

        for idx, (player, opening) in enumerate(player_opening_pairs):
            i = player_to_idx[player]
            j = opening_to_idx[opening]
            weight = 1.0 if weights is None else weights[idx]
            M[i, j] += weight

        # Calculate fitness and complexity
        F, Q = self.calculate(M)

        # Convert back to dictionaries
        player_fitness = {player: F[player_to_idx[player]] for player in players}
        opening_complexity = {opening: Q[opening_to_idx[opening]] for opening in openings}

        return player_fitness, opening_complexity


def calculate_diversity_score(bipartite_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate diversity scores for players and openings.

    Player diversity: Number of distinct openings played
    Opening diversity: Number of distinct players who used it

    Args:
        bipartite_matrix: Binary or weighted matrix (n_players, n_openings)

    Returns:
        Tuple of (player_diversity, opening_diversity)
    """
    # Player diversity: count non-zero entries per row
    player_diversity = np.count_nonzero(bipartite_matrix, axis=1)

    # Opening diversity: count non-zero entries per column
    opening_diversity = np.count_nonzero(bipartite_matrix, axis=0)

    return player_diversity, opening_diversity


def calculate_player_success(
    bipartite_matrix: np.ndarray,
    player_ratings: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Calculate player success metric.

    If ratings available, use normalized ratings.
    Otherwise, use opening diversity as proxy for success.

    Args:
        bipartite_matrix: Binary or weighted matrix (n_players, n_openings)
        player_ratings: Optional array of player ratings

    Returns:
        Array of player success scores
    """
    if player_ratings is not None:
        # Normalize ratings to [0, 1] range
        min_rating = np.min(player_ratings)
        max_rating = np.max(player_ratings)
        if max_rating > min_rating:
            return (player_ratings - min_rating) / (max_rating - min_rating)
        else:
            return np.ones_like(player_ratings)
    else:
        # Use diversity as proxy for success
        player_diversity, _ = calculate_diversity_score(bipartite_matrix)
        max_diversity = np.max(player_diversity)
        if max_diversity > 0:
            return player_diversity / max_diversity
        else:
            return np.ones(bipartite_matrix.shape[0])
