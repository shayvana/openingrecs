import networkx as nx
import matplotlib.pyplot as plt
import pickle

def visualize_bipartite_network(bipartite_network_path):
    with open(bipartite_network_path, 'rb') as f:
        B = pickle.load(f)
    
    pos = {}
    for node, data in B.nodes(data=True):
        if data['bipartite'] == 0:
            pos[node] = (1, len(pos))
        else:
            pos[node] = (2, len(pos))

    plt.figure(figsize=(12, 8))
    nx.draw(B, pos, with_labels=True, node_size=50, node_color="skyblue", font_size=8)
    plt.title('Bipartite Network of Players and Openings')
    plt.show()

if __name__ == "__main__":
    bipartite_network_path = 'data/bipartite_network.gpickle'
    visualize_bipartite_network(bipartite_network_path)
