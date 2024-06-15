import chess.pgn
import networkx as nx
import pickle
from tqdm import tqdm

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

def build_bipartite_network(pgn_file):
    B = nx.Graph()

    with open(pgn_file) as pgn:
        for _ in tqdm(range(100000), desc="Reading and processing games"):
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            if game.headers["White"] and game.headers["Black"]:
                white_player = game.headers["White"]
                black_player = game.headers["Black"]
                if "Opening" in game.headers:
                    opening = clean_opening(game.headers["Opening"])
                    B.add_node(white_player, bipartite=0)
                    B.add_node(black_player, bipartite=0)
                    B.add_node(opening, bipartite=1)
                    B.add_edge(white_player, opening)
                    B.add_edge(black_player, opening)
    
    with open('data/bipartite_network.pkl', 'wb') as f:
        pickle.dump(B, f)

    return B

if __name__ == "__main__":
    bipartite_network = build_bipartite_network('data/limited_lichess_games.pgn')
