import networkx as nx
import numpy as np
import pickle
from bicm import BipartiteGraph
from tqdm import tqdm

def clean_opening(opening):
    # Remove everything after a colon and (White)/(Black)
    opening = opening.split(':')[0].strip()
    if '(White)' in opening:
        opening = opening.replace('(White)', '').strip()
    if '(Black)' in opening:
        opening = opening.replace('(Black)', '').strip()
    return opening

def project_bipartite_network(B):
    players = [n for n, d in B.nodes(data=True) if d['bipartite'] == 0]
    openings = [clean_opening(n) for n, d in B.nodes(data=True) if d['bipartite'] == 1]

    M = np.zeros((len(players), len(openings)))
    player_index = {player: i for i, player in enumerate(players)}
    opening_index = {opening: i for i, opening in enumerate(openings)}

    for player, opening in B.edges():
        if player in player_index and opening in opening_index:
            i = player_index[player]
            j = opening_index[opening]
            M[i, j] = 1

    return M, openings

def filter_network(W_star, M):
    W_filtered = np.zeros_like(W_star)
    player_degrees = np.sum(M, axis=1)
    opening_degrees = np.sum(M, axis=0)
    graph = BipartiteGraph(biadjacency=M)
    graph.solve_tool()

    p_vals = graph.get_weighted_pvals_mat()
    significance_level = 0.05

    for i in range(W_star.shape[0]):
        for j in range(W_star.shape[1]):
            if W_star[i, j] > 0 and p_vals[i, j] < significance_level:
                W_filtered[i, j] = W_star[i, j]

    return W_filtered

def build_filtered_relatedness_network(bipartite_network_path):
    print("Starting to build filtered relatedness network...")
    with open(bipartite_network_path, 'rb') as f:
        B = pickle.load(f)

    print("Projecting bipartite network...")
    M, openings = project_bipartite_network(B)

    print("Calculating co-occurrence matrix...")
    W_star = M.T @ M

    print("Filtering co-occurrence matrix...")
    W_filtered = filter_network(W_star, M)

    G = nx.Graph()

    print("Building relatedness network...")
    for i in tqdm(range(W_filtered.shape[0]), desc="Building edges"):
        for j in range(i + 1, W_filtered.shape[1]):
            if W_filtered[i, j] > 0:
                G.add_edge(openings[i], openings[j], weight=W_filtered[i, j])

    print("Checking if network is connected...")
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        print(f"The relatedness network has {len(components)} components. Merging components...")

        largest_component = components[0]
        for component in components[1:]:
            node_from_largest = next(iter(largest_component))
            node_from_component = next(iter(component))
            G.add_edge(node_from_largest, node_from_component, weight=0.01)
            largest_component.update(component)

    return G

if __name__ == "__main__":
    relatedness_network = build_filtered_relatedness_network('data/bipartite_network.pkl')
    with open('data/relatedness_network.pkl', 'wb') as f:
        pickle.dump(relatedness_network, f)
