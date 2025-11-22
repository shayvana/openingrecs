"""
Statistical Filtering for Opening Relatedness Networks

Implements z-score based filtering using the Bipartite Configuration Model (BiCM)
as described in the Nature paper.

The key difference from incorrect implementations:
- Uses Z-SCORES for filtering, not p-values
- Z-score = (observed - expected) / std_deviation
- Filter edges where z-score > threshold (typically 2 or 3)

References:
    - Saracco, F., et al. (2015). "Randomizing bipartite networks: the case of the World Trade Web"
    - Nature paper Section: "Filtered Relatedness Network"
"""

import numpy as np
import networkx as nx
from bicm import BipartiteGraph
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ZScoreFilter:
    """
    Filter co-occurrence network using z-scores from BiCM null model.

    The Bipartite Configuration Model (BiCM) generates a null model that
    preserves the degree sequences of the original bipartite network.
    We compare the observed co-occurrences against this null model using z-scores.
    """

    def __init__(
        self,
        z_threshold: float = 2.0,
        method: str = 'all'
    ):
        """
        Initialize z-score filter.

        Args:
            z_threshold: Minimum z-score to retain an edge (default: 2.0 = 2 sigma)
                        Common values: 2.0 (95% CI), 3.0 (99.7% CI)
            method: Method for calculating significance
                   'all' - use all z-scores
                   'positive' - only consider positive z-scores (observed > expected)
        """
        self.z_threshold = z_threshold
        self.method = method

    def filter_network(
        self,
        bipartite_matrix: np.ndarray,
        co_occurrence_matrix: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Filter co-occurrence network using z-scores.

        Args:
            bipartite_matrix: Binary or weighted bipartite adjacency matrix
                            Shape: (n_players, n_openings)
            co_occurrence_matrix: Optional pre-computed co-occurrence matrix
                                If None, will compute from bipartite_matrix

        Returns:
            Tuple of:
            - filtered_matrix: Filtered co-occurrence matrix
            - z_scores: Full z-score matrix
            - stats: Dictionary with filtering statistics

        Raises:
            ValueError: If matrices are invalid
        """
        logger.info("Starting z-score based filtering")

        # Validate input
        if bipartite_matrix.shape[0] == 0 or bipartite_matrix.shape[1] == 0:
            raise ValueError("Bipartite matrix cannot be empty")

        n_players, n_openings = bipartite_matrix.shape

        # Compute co-occurrence matrix if not provided
        if co_occurrence_matrix is None:
            logger.debug("Computing co-occurrence matrix")
            co_occurrence_matrix = self._compute_co_occurrence(bipartite_matrix)

        # Convert to binary matrix for BiCM (it expects binary)
        binary_matrix = (bipartite_matrix > 0).astype(int)

        # Initialize BiCM model
        logger.debug("Initializing BiCM model")
        try:
            bicm = BipartiteGraph(biadjacency=binary_matrix)
            bicm.solve_tool()
        except Exception as e:
            logger.error(f"BiCM initialization failed: {e}")
            raise ValueError(f"Failed to initialize BiCM model: {e}")

        # Get expected values and standard deviations for co-occurrences
        logger.debug("Computing expected co-occurrences under null model")
        expected_matrix, std_matrix = self._compute_expected_cooccurrence(
            bicm,
            n_openings
        )

        # Calculate z-scores
        # z = (observed - expected) / std
        logger.debug("Computing z-scores")
        with np.errstate(divide='ignore', invalid='ignore'):
            z_scores = np.zeros_like(co_occurrence_matrix, dtype=float)
            valid_mask = std_matrix > 0
            z_scores[valid_mask] = (
                (co_occurrence_matrix[valid_mask] - expected_matrix[valid_mask])
                / std_matrix[valid_mask]
            )
            # Replace invalid values with 0
            z_scores[~np.isfinite(z_scores)] = 0

        # Filter based on z-scores
        if self.method == 'positive':
            # Only keep edges where observed > expected AND z-score > threshold
            filter_mask = (z_scores > self.z_threshold) & (co_occurrence_matrix > expected_matrix)
        else:
            # Keep edges where |z-score| > threshold
            filter_mask = np.abs(z_scores) > self.z_threshold

        filtered_matrix = np.where(filter_mask, co_occurrence_matrix, 0)

        # Compute statistics
        total_possible = (n_openings * (n_openings - 1)) // 2  # Excluding diagonal
        observed_edges = np.count_nonzero(np.triu(co_occurrence_matrix, k=1))
        filtered_edges = np.count_nonzero(np.triu(filtered_matrix, k=1))

        stats = {
            'total_possible_edges': total_possible,
            'observed_edges': observed_edges,
            'filtered_edges': filtered_edges,
            'retention_rate': filtered_edges / observed_edges if observed_edges > 0 else 0,
            'z_threshold': self.z_threshold,
            'method': self.method,
            'mean_z_score': np.mean(z_scores[np.triu_indices_from(z_scores, k=1)]),
            'max_z_score': np.max(z_scores),
            'min_z_score': np.min(z_scores)
        }

        logger.info(
            f"Filtering complete: {observed_edges} -> {filtered_edges} edges "
            f"({stats['retention_rate']:.2%} retained)"
        )

        return filtered_matrix, z_scores, stats

    def _compute_co_occurrence(self, bipartite_matrix: np.ndarray) -> np.ndarray:
        """
        Compute co-occurrence matrix from bipartite matrix.

        Co-occurrence W*[i,j] = number of players who played both opening i and j
        W* = M^T @ M

        Args:
            bipartite_matrix: Shape (n_players, n_openings)

        Returns:
            Co-occurrence matrix of shape (n_openings, n_openings)
        """
        co_occurrence = bipartite_matrix.T @ bipartite_matrix

        # Set diagonal to zero (opening doesn't co-occur with itself)
        np.fill_diagonal(co_occurrence, 0)

        return co_occurrence

    def _compute_expected_cooccurrence(
        self,
        bicm: BipartiteGraph,
        n_openings: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute expected co-occurrence matrix and standard deviation under BiCM null model.

        The BiCM provides expected probabilities for each edge in the bipartite network.
        We use these to compute expected co-occurrences in the projected network.

        Args:
            bicm: Fitted BipartiteGraph object
            n_openings: Number of opening nodes

        Returns:
            Tuple of (expected_matrix, std_matrix) both shape (n_openings, n_openings)
        """
        # Get expected probability matrix from BiCM
        # This is the probability that each edge exists in the null model
        prob_matrix = bicm.get_bicm_matrix()

        # Expected co-occurrence between opening i and j:
        # E[W*_ij] = Σ_k p_ki * p_kj
        # where p_ki is probability that player k plays opening i
        expected = prob_matrix.T @ prob_matrix
        np.fill_diagonal(expected, 0)

        # Standard deviation of co-occurrence:
        # Var[W*_ij] = Σ_k p_ki * p_kj * (1 - p_ki) * (1 - p_kj)
        # (assuming independence of edge realizations)
        variance = np.zeros((n_openings, n_openings))
        for i in range(n_openings):
            for j in range(n_openings):
                if i != j:
                    # Variance contribution from each player
                    p_i = prob_matrix[:, i]
                    p_j = prob_matrix[:, j]
                    variance[i, j] = np.sum(p_i * p_j * (1 - p_i) * (1 - p_j))

        std_matrix = np.sqrt(variance)

        return expected, std_matrix

    def filter_graph(
        self,
        bipartite_graph: nx.Graph,
        opening_bipartite_value: int = 1
    ) -> nx.Graph:
        """
        Filter a NetworkX bipartite graph and return projected filtered graph.

        Args:
            bipartite_graph: NetworkX bipartite graph
            opening_bipartite_value: Bipartite value for opening nodes (default: 1)

        Returns:
            Filtered NetworkX graph of openings with weighted edges
        """
        # Extract bipartite matrix
        players = [n for n, d in bipartite_graph.nodes(data=True)
                  if d.get('bipartite') != opening_bipartite_value]
        openings = [n for n, d in bipartite_graph.nodes(data=True)
                   if d.get('bipartite') == opening_bipartite_value]

        # Build adjacency matrix
        n_players = len(players)
        n_openings = len(openings)
        bipartite_matrix = np.zeros((n_players, n_openings))

        player_to_idx = {p: i for i, p in enumerate(players)}
        opening_to_idx = {o: i for i, o in enumerate(openings)}

        for player, opening in bipartite_graph.edges():
            if player in player_to_idx and opening in opening_to_idx:
                i = player_to_idx[player]
                j = opening_to_idx[opening]
                bipartite_matrix[i, j] = 1
            elif opening in player_to_idx and player in opening_to_idx:
                # Edge could be (opening, player) instead
                i = player_to_idx[opening]
                j = opening_to_idx[player]
                bipartite_matrix[i, j] = 1

        # Filter the network
        filtered_matrix, z_scores, stats = self.filter_network(bipartite_matrix)

        logger.info(f"Network filtering stats: {stats}")

        # Build NetworkX graph from filtered matrix
        G = nx.Graph()

        # Add all opening nodes
        for opening in openings:
            G.add_node(opening)

        # Add filtered edges
        for i in range(n_openings):
            for j in range(i + 1, n_openings):  # Upper triangle only
                if filtered_matrix[i, j] > 0:
                    G.add_edge(
                        openings[i],
                        openings[j],
                        weight=float(filtered_matrix[i, j]),
                        z_score=float(z_scores[i, j])
                    )

        logger.info(f"Created filtered graph with {G.number_of_nodes()} nodes "
                   f"and {G.number_of_edges()} edges")

        return G


def analyze_network_components(graph: nx.Graph) -> dict:
    """
    Analyze connected components in a graph.

    IMPORTANT: According to the paper, disconnected components should NOT
    be artificially connected. This function reports the structure as-is.

    Args:
        graph: NetworkX graph

    Returns:
        Dictionary with component analysis
    """
    if not nx.is_connected(graph):
        components = list(nx.connected_components(graph))
        component_sizes = [len(c) for c in components]

        logger.warning(
            f"Graph has {len(components)} disconnected components. "
            f"Sizes: {sorted(component_sizes, reverse=True)[:10]}"
        )

        return {
            'is_connected': False,
            'num_components': len(components),
            'component_sizes': component_sizes,
            'largest_component_size': max(component_sizes),
            'largest_component_fraction': max(component_sizes) / graph.number_of_nodes()
        }
    else:
        return {
            'is_connected': True,
            'num_components': 1,
            'component_sizes': [graph.number_of_nodes()],
            'largest_component_size': graph.number_of_nodes(),
            'largest_component_fraction': 1.0
        }
