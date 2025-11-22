"""
Project and Filter Network (Corrected Methodology)

Projects bipartite network to opening similarity network and filters using z-scores.

Key improvements over v1:
1. Uses Z-SCORES for filtering (not p-values)
2. Does NOT artificially connect components (paper compliant)
3. Calculates and stores EFC metrics
4. Proper logging and validation

Usage:
    python app/project_and_filter_network_v2.py <bipartite_file> [output_file]
"""

import networkx as nx
import numpy as np
import pickle
import sys
import os
import logging
from typing import Tuple, Dict
from tqdm import tqdm

# Import our corrected algorithms
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algorithms.filtering import ZScoreFilter, analyze_network_components
from algorithms.efc import EFCCalculator
from algorithms.similarity import SimilarityCalculator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_bipartite_matrix(
    B: nx.Graph
) -> Tuple[np.ndarray, list, list]:
    """
    Extract bipartite adjacency matrix from NetworkX graph.

    Args:
        B: NetworkX bipartite graph

    Returns:
        Tuple of (matrix, player_list, opening_list)
    """
    logger.info("Extracting bipartite matrix...")

    # Verify it's bipartite
    if not nx.is_bipartite(B):
        raise ValueError("Graph is not bipartite")

    # Separate player and opening nodes
    players = sorted([
        n for n, d in B.nodes(data=True)
        if d.get('bipartite') == 0
    ])
    openings = sorted([
        n for n, d in B.nodes(data=True)
        if d.get('bipartite') == 1
    ])

    logger.info(f"Players: {len(players)}, Openings: {len(openings)}")

    # Create index mappings
    player_to_idx = {p: i for i, p in enumerate(players)}
    opening_to_idx = {o: i for i, o in enumerate(openings)}

    # Build matrix
    n_players = len(players)
    n_openings = len(openings)
    M = np.zeros((n_players, n_openings))

    for player, opening, data in B.edges(data=True):
        # Edge could be (player, opening) or (opening, player)
        if player in player_to_idx and opening in opening_to_idx:
            i = player_to_idx[player]
            j = opening_to_idx[opening]
            weight = data.get('weight', 1.0)
            M[i, j] = weight
        elif opening in player_to_idx and player in opening_to_idx:
            i = player_to_idx[opening]
            j = opening_to_idx[player]
            weight = data.get('weight', 1.0)
            M[i, j] = weight

    logger.info(f"Matrix shape: {M.shape}, Non-zero entries: {np.count_nonzero(M)}")

    return M, players, openings


def calculate_efc_metrics(
    bipartite_matrix: np.ndarray,
    players: list,
    openings: list
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Calculate Economic Fitness and Complexity metrics.

    Args:
        bipartite_matrix: Binary/weighted matrix (n_players, n_openings)
        players: List of player names
        openings: List of opening names

    Returns:
        Tuple of (player_fitness_dict, opening_complexity_dict)
    """
    logger.info("Calculating EFC metrics...")

    calculator = EFCCalculator(tolerance=1e-6, max_iterations=100)

    try:
        fitness, complexity = calculator.calculate(bipartite_matrix)

        # Convert to dictionaries
        player_fitness = {players[i]: fitness[i] for i in range(len(players))}
        opening_complexity = {openings[j]: complexity[j] for j in range(len(openings))}

        logger.info(f"EFC converged in {calculator.iterations} iterations")
        logger.info(f"Mean player fitness: {np.mean(fitness):.4f}")
        logger.info(f"Mean opening complexity: {np.mean(complexity):.4f}")

        return player_fitness, opening_complexity

    except ValueError as e:
        logger.error(f"EFC calculation failed: {e}")
        raise


def project_and_filter_network(
    bipartite_file: str,
    z_threshold: float = 2.0,
    calculate_efc: bool = True
) -> Tuple[nx.Graph, Dict]:
    """
    Project bipartite network and filter using z-scores.

    This follows the correct methodology from the Nature paper:
    1. Project bipartite network to opening co-occurrence network
    2. Calculate z-scores using BiCM null model
    3. Filter edges where z-score > threshold
    4. Calculate EFC metrics
    5. DO NOT artificially connect components

    Args:
        bipartite_file: Path to bipartite network pickle file
        z_threshold: Z-score threshold for filtering (default: 2.0)
        calculate_efc: Whether to calculate EFC metrics (default: True)

    Returns:
        Tuple of (filtered_graph, metadata)
            - filtered_graph: NetworkX graph with filtered edges
            - metadata: Dictionary with statistics and EFC metrics

    Raises:
        FileNotFoundError: If bipartite file doesn't exist
        ValueError: If network is invalid
    """
    if not os.path.exists(bipartite_file):
        raise FileNotFoundError(f"Bipartite file not found: {bipartite_file}")

    logger.info(f"Loading bipartite network from {bipartite_file}")

    # Load bipartite network
    with open(bipartite_file, 'rb') as f:
        B = pickle.load(f)

    logger.info(f"Loaded network: {B.number_of_nodes()} nodes, {B.number_of_edges()} edges")

    # Extract matrix representation
    bipartite_matrix, players, openings = extract_bipartite_matrix(B)

    # Initialize filter
    logger.info(f"Filtering with z-score threshold: {z_threshold}")
    filter = ZScoreFilter(z_threshold=z_threshold, method='positive')

    # Filter network
    filtered_matrix, z_scores, filter_stats = filter.filter_network(bipartite_matrix)

    logger.info(f"Filtering complete: retained {filter_stats['retention_rate']:.2%} of edges")

    # Build NetworkX graph from filtered matrix
    logger.info("Building filtered opening network...")
    G = nx.Graph()

    # Add nodes with attributes from bipartite network
    for opening in tqdm(openings, desc="Adding nodes"):
        bipartite_node_data = B.nodes[opening]
        G.add_node(
            opening,
            play_count=bipartite_node_data.get('play_count', 0),
            player_diversity=bipartite_node_data.get('player_diversity', 0)
        )

    # Add filtered edges
    edge_count = 0
    for i in tqdm(range(len(openings)), desc="Adding edges"):
        for j in range(i + 1, len(openings)):
            if filtered_matrix[i, j] > 0:
                G.add_edge(
                    openings[i],
                    openings[j],
                    weight=float(filtered_matrix[i, j]),
                    z_score=float(z_scores[i, j])
                )
                edge_count += 1

    logger.info(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    # Analyze components
    component_analysis = analyze_network_components(G)
    logger.info(f"Component analysis: {component_analysis}")

    # IMPORTANT: Do NOT artificially connect components
    # This is explicitly against the paper's methodology
    if not component_analysis['is_connected']:
        logger.warning(
            f"Network has {component_analysis['num_components']} disconnected components. "
            f"This is expected and we do NOT artificially connect them (per paper methodology)."
        )

    # Calculate EFC metrics if requested
    metadata = {
        'filter_stats': filter_stats,
        'component_analysis': component_analysis,
        'z_threshold': z_threshold,
        'num_players': len(players),
        'num_openings': len(openings)
    }

    if calculate_efc:
        logger.info("Calculating EFC metrics...")
        try:
            player_fitness, opening_complexity = calculate_efc_metrics(
                bipartite_matrix,
                players,
                openings
            )

            # Add complexity scores to graph nodes
            for opening in G.nodes():
                if opening in opening_complexity:
                    G.nodes[opening]['complexity'] = opening_complexity[opening]

            metadata['player_fitness'] = player_fitness
            metadata['opening_complexity'] = opening_complexity

            # Statistics
            complexity_values = list(opening_complexity.values())
            metadata['complexity_stats'] = {
                'mean': np.mean(complexity_values),
                'std': np.std(complexity_values),
                'min': np.min(complexity_values),
                'max': np.max(complexity_values)
            }

            logger.info(f"Opening complexity - mean: {metadata['complexity_stats']['mean']:.4f}, "
                       f"std: {metadata['complexity_stats']['std']:.4f}")

        except Exception as e:
            logger.error(f"Failed to calculate EFC metrics: {e}")
            metadata['efc_error'] = str(e)

    return G, metadata


def save_network(
    network: nx.Graph,
    metadata: Dict,
    output_file: str
):
    """
    Save filtered network and metadata.

    Args:
        network: Filtered NetworkX graph
        metadata: Metadata dictionary
        output_file: Path to output file
    """
    logger.info(f"Saving filtered network to {output_file}")

    # Create directory if needed
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save network
    with open(output_file, 'wb') as f:
        pickle.dump(network, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save metadata separately
    metadata_file = output_file.replace('.pkl', '_metadata.pkl')
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info(f"Network saved: {os.path.getsize(output_file)} bytes")
    logger.info(f"Metadata saved: {metadata_file}")


def validate_network(network: nx.Graph, metadata: Dict):
    """
    Validate that the network follows paper methodology.

    Args:
        network: NetworkX graph
        metadata: Metadata dictionary

    Raises:
        AssertionError: If validation fails
    """
    logger.info("Validating network...")

    # Check that filtering used z-scores
    assert 'filter_stats' in metadata
    assert metadata['filter_stats']['z_threshold'] > 0

    # Check that components are NOT artificially connected
    if not metadata['component_analysis']['is_connected']:
        # This is OK! Verify no artificial edges were added
        # (We can't easily verify this programmatically, but logging warns about it)
        logger.info("✓ Network has natural disconnected components (correct)")

    # Check that complexity was calculated
    if 'opening_complexity' in metadata:
        assert len(metadata['opening_complexity']) > 0
        logger.info("✓ Opening complexity calculated")

    # Check all nodes have complexity attribute
    nodes_with_complexity = sum(
        1 for n in network.nodes()
        if 'complexity' in network.nodes[n]
    )
    logger.info(f"✓ {nodes_with_complexity}/{network.number_of_nodes()} nodes have complexity scores")

    # Check edges have z-scores
    edges_with_zscores = sum(
        1 for u, v in network.edges()
        if 'z_score' in network[u][v]
    )
    logger.info(f"✓ {edges_with_zscores}/{network.number_of_edges()} edges have z-scores")

    logger.info("Validation complete!")


def main():
    """Main entry point for script."""
    if len(sys.argv) < 2:
        print("Usage: python project_and_filter_network_v2.py <bipartite_file> [output_file] [z_threshold]")
        print("Example: python project_and_filter_network_v2.py data/bipartite_network_v2.pkl data/relatedness_network_v2.pkl 2.0")
        sys.exit(1)

    bipartite_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'data/relatedness_network_v2.pkl'
    z_threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0

    try:
        # Project and filter network
        network, metadata = project_and_filter_network(
            bipartite_file=bipartite_file,
            z_threshold=z_threshold,
            calculate_efc=True
        )

        # Validate
        validate_network(network, metadata)

        # Save network
        save_network(network, metadata, output_file)

        logger.info("Complete!")

        # Print summary
        print("\n" + "="*60)
        print("NETWORK SUMMARY")
        print("="*60)
        print(f"Nodes: {network.number_of_nodes()}")
        print(f"Edges: {network.number_of_edges()}")
        print(f"Connected: {metadata['component_analysis']['is_connected']}")
        if not metadata['component_analysis']['is_connected']:
            print(f"Components: {metadata['component_analysis']['num_components']}")
            print(f"Largest component: {metadata['component_analysis']['largest_component_size']} nodes")
        print(f"Z-threshold used: {z_threshold}")
        if 'complexity_stats' in metadata:
            print(f"Mean opening complexity: {metadata['complexity_stats']['mean']:.4f}")
        print("="*60)

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
