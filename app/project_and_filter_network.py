import networkx as nx
import numpy as np
import pickle
from bicm import BipartiteGraph
from tqdm import tqdm

def clean_opening(opening):
    # Remove everything after a colon and (White)/(Black) suffix
    return opening.split(":")[0].replace("(White)", "").replace("(Black)", "").strip()

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
    graph = BipartiteGraph(biadjacency=M)
    graph.solve_tool()

    p_vals = graph.get_weighted_pvals_mat()
    significance_level = 0.05

    W_filtered = np.zeros_like(W_star)
    for i in range(W_star.shape[0]):
        for j in range(W_star.shape[1]):
            if W_star[i, j] > 0 and p_vals[i, j] < significance_level:
                W_filtered[i, j] = W_star[i, j]

    return W_filtered

def build_filtered_relatedness_network(bipartite_network_path):
    print("Starting to build filtered relatedness network...")
    print("Loading bipartite network...")
    with open(bipartite_network_path, 'rb') as f:
        B = pickle.load(f)

    print("Projecting bipartite network to unipartite network...")
    print("Projecting bipartite network...")
    M, openings = project_bipartite_network(B)
    print("Calculating co-occurrence matrix...")
    W_star = M.T @ M

    print("Filtering co-occurrence matrix...")
    W_filtered = filter_network(W_star, M)

    print("Building graph from filtered co-occurrence matrix...")
    G = nx.Graph()
    for i in range(W_filtered.shape[0]):
        for j in range(W_filtered.shape[1]):
            if W_filtered[i, j] > 0:
                G.add_edge(openings[i], openings[j], weight=W_filtered[i, j])

    components = list(nx.connected_components(G))
    print(f"The relatedness network has {len(components)} components. Merging components...")
    if len(components) > 1:
        for i, component in enumerate(components[1:], start=1):
            G.add_edge(list(components[0])[0], list(component)[0], weight=1.0)  # Minimal edge to connect components

    return G

if __name__ == "__main__":
    relatedness_network = build_filtered_relatedness_network('data/bipartite_network.pkl')
    with open('data/relatedness_network.pkl', 'wb') as f:
        pickle.dump(relatedness_network, f)
