import networkx as nx
import numpy as np
import pickle
from bicm import BiCM

def clean_opening(opening):
    # Remove everything after a colon
    if ':' in opening:
        opening = opening.split(':')[0].strip()
    # Remove (White) and (Black) from the end of the string
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
    bicm_model = BiCM(player_degrees, opening_degrees)

    for i in range(W_star.shape[0]):
        for j in range(W_star.shape[1]):
            if W_star[i, j] > 0:
                p_value = bicm_model.get_p_value(W_star[i, j], i, j)
                if p_value < 0.05:
                    W_filtered[i, j] = W_star[i, j]

    return W_filtered

def build_filtered_relatedness_network(bipartite_network_path):
    with open(bipartite_network_path, 'rb') as f:
        B = pickle.load(f)
    M, openings = project_bipartite_network(B)
    W_star = M.T @ M
    W_filtered = filter_network(W_star, M)

    G = nx.Graph()

    for i in range(W_filtered.shape[0]):
        for j in range(W_filtered.shape[1]):
            if W_filtered[i, j] > 0:
                G.add_edge(openings[i], openings[j], weight=W_filtered[i, j])

    return G

relatedness_network = build_filtered_relatedness_network('data/bipartite_network.pkl')
with open('data/relatedness_network.pkl', 'wb') as f:
    pickle.dump(relatedness_network, f)
