import networkx as nx
import matplotlib.pyplot as plt
import pickle

def load_relatedness_network(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def main():
    relatedness_network = load_relatedness_network('data/relatedness_network.pkl')
    
    # Filter out low-weight edges to reduce clutter
    threshold = 0.5
    filtered_edges = [(u, v) for u, v, d in relatedness_network.edges(data=True) if d['weight'] >= threshold]
    filtered_network = nx.Graph()
    filtered_network.add_edges_from(filtered_edges)
    
    # Identify and plot disconnected components separately
    components = [filtered_network.subgraph(c).copy() for c in nx.connected_components(filtered_network)]

    plt.figure(figsize=(20, 20))
    
    for i, component in enumerate(components):
        pos = nx.spring_layout(component, k=0.1, iterations=50)
        nx.draw_networkx_nodes(component, pos, node_color='blue', node_size=50)
        nx.draw_networkx_edges(component, pos, alpha=0.3)
        nx.draw_networkx_labels(component, pos, font_size=8, font_color='black', font_family='sans-serif')
    
    plt.title('Filtered Relatedness Network of Chess Openings')
    plt.show()

if __name__ == "__main__":
    main()
