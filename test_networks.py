import unittest
import networkx as nx
import pickle

class TestBipartiteNetwork(unittest.TestCase):

    def setUp(self):
        with open('data/bipartite_network.pkl', 'rb') as f:
            self.bipartite_network = pickle.load(f)
    
    def test_is_bipartite(self):
        is_bipartite = nx.is_bipartite(self.bipartite_network)
        self.assertTrue(is_bipartite, "The network should be bipartite.")

    def test_nodes_have_bipartite_attribute(self):
        for node, data in self.bipartite_network.nodes(data=True):
            self.assertIn('bipartite', data, f"Node {node} should have a 'bipartite' attribute.")
    
    def test_node_bipartite_values(self):
        for node, data in self.bipartite_network.nodes(data=True):
            self.assertIn(data['bipartite'], [0, 1], f"Node {node} should have 'bipartite' value of 0 or 1.")

class TestBipartiteNetworkCategorization(unittest.TestCase):

    def setUp(self):
        with open('data/bipartite_network.pkl', 'rb') as f:
            self.bipartite_network = pickle.load(f)
    
    def test_player_nodes(self):
        players = [n for n, d in self.bipartite_network.nodes(data=True) if d['bipartite'] == 0]
        for player in players:
            self.assertIsInstance(player, str, f"Player node {player} should be a string.")
    
    def test_opening_nodes(self):
        openings = [n for n, d in self.bipartite_network.nodes(data=True) if d['bipartite'] == 1]
        for opening in openings:
            self.assertIsInstance(opening, str, f"Opening node {opening} should be a string.")

class TestRelatednessNetwork(unittest.TestCase):

    def setUp(self):
        with open('data/relatedness_network.pkl', 'rb') as f:
            self.relatedness_network = pickle.load(f)

    def test_relatedness_network_edges(self):
        for u, v, data in self.relatedness_network.edges(data=True):
            self.assertIn('weight', data, f"Edge ({u}, {v}) should have a 'weight' attribute.")
    
    def test_no_self_loops(self):
        self_loops = list(nx.selfloop_edges(self.relatedness_network))
        self.assertEqual(len(self_loops), 0, "The relatedness network should not have self-loops.")

class TestRelatednessNetworkConnectivity(unittest.TestCase):

    def setUp(self):
        with open('data/relatedness_network.pkl', 'rb') as f:
            self.relatedness_network = pickle.load(f)

    def test_is_connected(self):
        is_connected = nx.is_connected(self.relatedness_network)
        self.assertTrue(is_connected, "The relatedness network should be connected.")

if __name__ == '__main__':
    unittest.main()



