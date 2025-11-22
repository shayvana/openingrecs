"""
Build Bipartite Network (Corrected Methodology)

Builds a bipartite network of players and openings following the Nature paper methodology.

Key improvements over v1:
1. Preserves player ratings for weighting
2. Stores edge weights (game counts)
3. Includes proper metadata
4. No premature normalization
5. Proper error handling and logging

Usage:
    python app/build_bipartite_network_v2.py <pgn_file> [output_file]
"""

import chess.pgn
import networkx as nx
import pickle
from tqdm import tqdm
from typing import Optional, Dict, Set
from collections import defaultdict
import logging
import sys
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clean_opening(opening: str) -> str:
    """
    Clean opening name while preserving main variation information.

    Rules:
    - Remove text after colon (sub-variations)
    - Remove (White) and (Black) suffixes
    - Preserve main opening name and variation

    Args:
        opening: Raw opening name from PGN

    Returns:
        Cleaned opening name

    Examples:
        "Sicilian Defense: Najdorf Variation" -> "Sicilian Defense"
        "French Defense (White)" -> "French Defense"
    """
    if ':' in opening:
        opening = opening.split(':')[0].strip()

    opening = opening.replace('(White)', '').replace('(Black)', '').strip()

    return opening


def extract_player_rating(headers: chess.pgn.Headers, color: str) -> Optional[int]:
    """
    Extract player rating from game headers.

    Args:
        headers: PGN headers dictionary
        color: 'White' or 'Black'

    Returns:
        Rating as integer, or None if not available
    """
    rating_key = f"{color}Elo"

    if rating_key in headers:
        try:
            rating = int(headers[rating_key])
            # Sanity check: typical chess ratings are 0-3500
            if 0 <= rating <= 3500:
                return rating
        except (ValueError, TypeError):
            pass

    return None


def build_bipartite_network(
    pgn_file: str,
    max_games: Optional[int] = None,
    min_rating: int = 0
) -> nx.Graph:
    """
    Build bipartite network from PGN file following paper methodology.

    Network structure:
    - Player nodes (bipartite=0): Represent players
        - Attributes: rating, game_count, opening_diversity
    - Opening nodes (bipartite=1): Represent openings
        - Attributes: play_count, player_diversity
    - Edges: Weighted by number of times player used opening
        - Attributes: weight, avg_rating

    Args:
        pgn_file: Path to PGN file
        max_games: Maximum number of games to process (None = all)
        min_rating: Minimum player rating to include (0 = include all)

    Returns:
        NetworkX bipartite graph

    Raises:
        FileNotFoundError: If PGN file doesn't exist
        ValueError: If PGN file is empty or invalid
    """
    if not os.path.exists(pgn_file):
        raise FileNotFoundError(f"PGN file not found: {pgn_file}")

    logger.info(f"Building bipartite network from {pgn_file}")
    logger.info(f"Max games: {max_games if max_games else 'unlimited'}")
    logger.info(f"Min rating: {min_rating}")

    B = nx.Graph()

    # Track statistics
    player_ratings: Dict[str, list] = defaultdict(list)  # player -> [ratings]
    player_openings: Dict[str, Set[str]] = defaultdict(set)  # player -> {openings}
    opening_players: Dict[str, Set[str]] = defaultdict(set)  # opening -> {players}
    edge_counts: Dict[tuple, int] = defaultdict(int)  # (player, opening) -> count
    edge_ratings: Dict[tuple, list] = defaultdict(list)  # (player, opening) -> [ratings]

    games_processed = 0
    games_with_opening = 0
    games_skipped_rating = 0

    with open(pgn_file) as pgn:
        progress_bar = tqdm(
            total=max_games if max_games else None,
            desc="Processing games"
        )

        while True:
            if max_games and games_processed >= max_games:
                break

            game = chess.pgn.read_game(pgn)
            if game is None:
                break

            games_processed += 1
            progress_bar.update(1)

            # Extract player information
            white_player = game.headers.get("White")
            black_player = game.headers.get("Black")
            opening = game.headers.get("Opening")

            if not white_player or not black_player or not opening:
                continue

            # Extract ratings
            white_rating = extract_player_rating(game.headers, "White")
            black_rating = extract_player_rating(game.headers, "Black")

            # Apply minimum rating filter
            if min_rating > 0:
                if not white_rating or white_rating < min_rating:
                    games_skipped_rating += 1
                    continue
                if not black_rating or black_rating < min_rating:
                    games_skipped_rating += 1
                    continue

            # Clean opening name
            opening_clean = clean_opening(opening)

            games_with_opening += 1

            # Process white player
            if white_rating:
                player_ratings[white_player].append(white_rating)
            player_openings[white_player].add(opening_clean)
            opening_players[opening_clean].add(white_player)
            edge_counts[(white_player, opening_clean)] += 1
            if white_rating:
                edge_ratings[(white_player, opening_clean)].append(white_rating)

            # Process black player
            if black_rating:
                player_ratings[black_player].append(black_rating)
            player_openings[black_player].add(opening_clean)
            opening_players[opening_clean].add(black_player)
            edge_counts[(black_player, opening_clean)] += 1
            if black_rating:
                edge_ratings[(black_player, opening_clean)].append(black_rating)

        progress_bar.close()

    logger.info(f"Processed {games_processed} games")
    logger.info(f"Games with opening info: {games_with_opening}")
    logger.info(f"Games skipped (rating filter): {games_skipped_rating}")
    logger.info(f"Unique players: {len(player_openings)}")
    logger.info(f"Unique openings: {len(opening_players)}")

    # Build NetworkX graph with attributes
    logger.info("Building NetworkX graph...")

    # Add player nodes
    for player in tqdm(player_openings.keys(), desc="Adding player nodes"):
        # Calculate average rating
        ratings = player_ratings.get(player, [])
        avg_rating = sum(ratings) / len(ratings) if ratings else None

        # Calculate diversity
        diversity = len(player_openings[player])

        # Calculate total games
        game_count = sum(
            edge_counts[(player, opening)]
            for opening in player_openings[player]
        )

        B.add_node(
            player,
            bipartite=0,
            node_type='player',
            rating=avg_rating,
            game_count=game_count,
            opening_diversity=diversity
        )

    # Add opening nodes
    for opening in tqdm(opening_players.keys(), desc="Adding opening nodes"):
        play_count = sum(
            edge_counts[(player, opening)]
            for player in opening_players[opening]
        )

        player_diversity = len(opening_players[opening])

        B.add_node(
            opening,
            bipartite=1,
            node_type='opening',
            play_count=play_count,
            player_diversity=player_diversity
        )

    # Add edges with weights
    logger.info("Adding edges...")
    for (player, opening), count in tqdm(edge_counts.items(), desc="Adding edges"):
        # Calculate average rating for this edge
        ratings = edge_ratings.get((player, opening), [])
        avg_rating = sum(ratings) / len(ratings) if ratings else None

        B.add_edge(
            player,
            opening,
            weight=count,
            avg_rating=avg_rating
        )

    # Verify bipartite structure
    if not nx.is_bipartite(B):
        logger.error("ERROR: Graph is not bipartite!")
        raise ValueError("Constructed graph is not bipartite")

    logger.info("Bipartite network construction complete")
    logger.info(f"Nodes: {B.number_of_nodes()} ({len(player_openings)} players + "
               f"{len(opening_players)} openings)")
    logger.info(f"Edges: {B.number_of_edges()}")

    # Report statistics
    player_nodes = [n for n, d in B.nodes(data=True) if d['bipartite'] == 0]
    opening_nodes = [n for n, d in B.nodes(data=True) if d['bipartite'] == 1]

    avg_player_degree = sum(B.degree(p) for p in player_nodes) / len(player_nodes)
    avg_opening_degree = sum(B.degree(o) for o in opening_nodes) / len(opening_nodes)

    logger.info(f"Average player degree: {avg_player_degree:.2f}")
    logger.info(f"Average opening degree: {avg_opening_degree:.2f}")

    return B


def save_network(network: nx.Graph, output_file: str):
    """
    Save network to pickle file.

    Args:
        network: NetworkX graph
        output_file: Path to output file
    """
    logger.info(f"Saving network to {output_file}")

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'wb') as f:
        pickle.dump(network, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info(f"Network saved successfully ({os.path.getsize(output_file)} bytes)")


def main():
    """Main entry point for script."""
    if len(sys.argv) < 2:
        print("Usage: python build_bipartite_network_v2.py <pgn_file> [output_file]")
        print("Example: python build_bipartite_network_v2.py data/games.pgn data/bipartite_network_v2.pkl")
        sys.exit(1)

    pgn_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'data/bipartite_network_v2.pkl'

    try:
        # Build network
        network = build_bipartite_network(
            pgn_file=pgn_file,
            max_games=None,  # Process all games
            min_rating=0     # Include all ratings
        )

        # Save network
        save_network(network, output_file)

        logger.info("Complete!")

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
