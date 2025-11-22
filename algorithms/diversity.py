"""
Opening Diversity Metrics

Implements diversity calculations for chess openings as described in the Nature paper.

Opening diversity measures:
1. Branching factor: Number of distinct continuations from opening positions
2. Move diversity: Number of unique variations played
3. Player diversity: Number of distinct players who use the opening

References:
    - Nature paper Section: "Opening Diversity"
"""

import numpy as np
import chess
import chess.pgn
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class DiversityCalculator:
    """
    Calculate diversity metrics for chess openings.

    Diversity can be measured in multiple ways:
    - Structural diversity: Number of possible moves/positions
    - Usage diversity: How many different variations are actually played
    - Player diversity: How many distinct players use the opening
    """

    def __init__(self, max_depth: int = 10):
        """
        Initialize diversity calculator.

        Args:
            max_depth: Maximum move depth to consider for opening analysis
                      (default: 10 moves for each side = 20 plies)
        """
        self.max_depth = max_depth

    def calculate_structural_diversity(
        self,
        opening_games: List[chess.pgn.Game]
    ) -> float:
        """
        Calculate structural diversity of an opening based on position tree.

        This measures the branching factor: number of distinct positions
        reached within the opening phase.

        Args:
            opening_games: List of games that start with this opening

        Returns:
            Diversity score based on unique positions
        """
        if not opening_games:
            return 0.0

        positions = set()

        for game in opening_games:
            board = game.board()
            move_count = 0

            for move in game.mainline_moves():
                if move_count >= self.max_depth:
                    break

                board.push(move)
                # Use FEN without move counters for position uniqueness
                fen = self._normalize_fen(board.fen())
                positions.add(fen)
                move_count += 1

        diversity = len(positions)
        logger.debug(f"Structural diversity: {diversity} unique positions")

        return diversity

    def calculate_move_diversity(
        self,
        opening_games: List[chess.pgn.Game]
    ) -> Dict[str, float]:
        """
        Calculate diversity of moves played at each position.

        Returns statistics about move diversity in the opening tree.

        Args:
            opening_games: List of games that start with this opening

        Returns:
            Dictionary with diversity metrics:
            - mean_branching_factor: Average number of different moves per position
            - max_branching_factor: Maximum branching at any position
            - total_unique_positions: Total distinct positions
            - average_depth: Average depth explored in the opening
        """
        if not opening_games:
            return {
                'mean_branching_factor': 0.0,
                'max_branching_factor': 0,
                'total_unique_positions': 0,
                'average_depth': 0.0
            }

        # Track moves played at each position
        position_moves = defaultdict(set)  # position -> set of moves
        position_counts = defaultdict(int)  # how often each position appears
        depths = []

        for game in opening_games:
            board = game.board()
            move_count = 0

            for move in game.mainline_moves():
                if move_count >= self.max_depth:
                    break

                fen = self._normalize_fen(board.fen())
                position_moves[fen].add(move.uci())
                position_counts[fen] += 1

                board.push(move)
                move_count += 1

            depths.append(move_count)

        # Calculate statistics
        branching_factors = [len(moves) for moves in position_moves.values()]

        metrics = {
            'mean_branching_factor': np.mean(branching_factors) if branching_factors else 0.0,
            'max_branching_factor': max(branching_factors) if branching_factors else 0,
            'total_unique_positions': len(position_moves),
            'average_depth': np.mean(depths) if depths else 0.0,
            'total_positions_visited': sum(position_counts.values())
        }

        logger.debug(f"Move diversity metrics: {metrics}")
        return metrics

    def calculate_variation_diversity(
        self,
        opening_name: str,
        games: List[chess.pgn.Game]
    ) -> int:
        """
        Calculate number of distinct variations within an opening.

        A variation is defined as a unique sequence of moves up to max_depth.

        Args:
            opening_name: Name of the opening
            games: List of games with this opening

        Returns:
            Number of unique variations
        """
        if not games:
            return 0

        variations = set()

        for game in games:
            move_sequence = []
            move_count = 0

            for move in game.mainline_moves():
                if move_count >= self.max_depth:
                    break
                move_sequence.append(move.uci())
                move_count += 1

            # Use tuple of moves as variation identifier
            variation = tuple(move_sequence)
            variations.add(variation)

        logger.debug(f"{opening_name}: {len(variations)} distinct variations")
        return len(variations)

    def calculate_complexity_from_games(
        self,
        games: List[chess.pgn.Game],
        include_tactical: bool = False
    ) -> Dict[str, float]:
        """
        Calculate comprehensive complexity metrics from games.

        Args:
            games: List of games
            include_tactical: Whether to include tactical complexity
                            (requires more computation)

        Returns:
            Dictionary with complexity scores
        """
        if not games:
            return {
                'structural_diversity': 0.0,
                'mean_branching': 0.0,
                'variation_count': 0,
                'average_game_length': 0.0
            }

        # Calculate various metrics
        structural_div = self.calculate_structural_diversity(games)
        move_div = self.calculate_move_diversity(games)
        variation_count = len(set(
            tuple(move.uci() for move in game.mainline_moves()[:self.max_depth])
            for game in games
        ))

        game_lengths = []
        for game in games:
            length = sum(1 for _ in game.mainline_moves())
            game_lengths.append(length)

        complexity = {
            'structural_diversity': structural_div,
            'mean_branching': move_div['mean_branching_factor'],
            'variation_count': variation_count,
            'average_game_length': np.mean(game_lengths) if game_lengths else 0.0,
            'total_positions': move_div['total_unique_positions']
        }

        return complexity

    def _normalize_fen(self, fen: str) -> str:
        """
        Normalize FEN string for position comparison.

        Removes move counters and castling rights to focus on piece positions.

        Args:
            fen: Full FEN string

        Returns:
            Normalized FEN (only piece positions and turn)
        """
        parts = fen.split()
        # Keep only: piece positions, active color, castling, en passant
        # Remove: halfmove clock, fullmove number
        return ' '.join(parts[:4])


def calculate_opening_popularity(
    opening_counts: Dict[str, int]
) -> Dict[str, float]:
    """
    Calculate popularity score for each opening.

    Popularity is the relative frequency of the opening being played.

    Args:
        opening_counts: Dictionary mapping opening names to play counts

    Returns:
        Dictionary mapping opening names to popularity scores (0-1)
    """
    if not opening_counts:
        return {}

    total_games = sum(opening_counts.values())

    popularity = {
        opening: count / total_games
        for opening, count in opening_counts.items()
    }

    return popularity


def calculate_player_diversity_per_opening(
    opening_players: Dict[str, Set[str]]
) -> Dict[str, int]:
    """
    Calculate how many distinct players used each opening.

    Higher player diversity indicates a more accessible/popular opening.

    Args:
        opening_players: Dictionary mapping opening names to sets of player names

    Returns:
        Dictionary mapping opening names to player diversity counts
    """
    return {
        opening: len(players)
        for opening, players in opening_players.items()
    }


def calculate_entropy(probabilities: np.ndarray) -> float:
    """
    Calculate Shannon entropy of a probability distribution.

    Higher entropy = more diverse/uncertain distribution.

    Args:
        probabilities: Array of probabilities (should sum to 1)

    Returns:
        Entropy in bits
    """
    # Remove zero probabilities to avoid log(0)
    probs = probabilities[probabilities > 0]

    if len(probs) == 0:
        return 0.0

    entropy = -np.sum(probs * np.log2(probs))
    return entropy


def calculate_opening_entropy(
    opening_move_distribution: Dict[str, int]
) -> float:
    """
    Calculate entropy of move distribution for an opening.

    Measures how "spread out" the move choices are in the opening.

    Args:
        opening_move_distribution: Dictionary mapping moves to frequencies

    Returns:
        Entropy score (higher = more diverse move choices)
    """
    if not opening_move_distribution:
        return 0.0

    counts = np.array(list(opening_move_distribution.values()))
    probabilities = counts / counts.sum()

    return calculate_entropy(probabilities)


def aggregate_diversity_metrics(
    structural_diversity: float,
    player_diversity: int,
    variation_count: int,
    popularity: float,
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Aggregate multiple diversity metrics into a single score.

    Args:
        structural_diversity: Number of unique positions
        player_diversity: Number of distinct players
        variation_count: Number of variations
        popularity: Relative frequency of play
        weights: Optional weights for each metric

    Returns:
        Aggregated diversity score
    """
    if weights is None:
        # Default weights
        weights = {
            'structural': 0.3,
            'player': 0.3,
            'variation': 0.2,
            'popularity': 0.2
        }

    # Normalize each metric to [0, 1] range (approximate)
    # These are heuristic normalizations
    normalized_structural = min(structural_diversity / 100, 1.0)
    normalized_player = min(player_diversity / 1000, 1.0)
    normalized_variation = min(variation_count / 50, 1.0)
    normalized_popularity = popularity  # Already in [0, 1]

    # Weighted sum
    score = (
        weights['structural'] * normalized_structural +
        weights['player'] * normalized_player +
        weights['variation'] * normalized_variation +
        weights['popularity'] * normalized_popularity
    )

    return score
