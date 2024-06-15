from collections import defaultdict
import numpy as np
from scipy.stats import zscore

def build_player_opening_network(openings):
    network = defaultdict(list)
    for opening in openings:
        network[opening].append(1)
    return network

def project_opening_network(network):
    openings = list(network.keys())
    size = len(openings)
    co_occurrence = np.zeros((size, size))
    for i, opening1 in enumerate(openings):
        for j, opening2 in enumerate(openings):
            if i != j:
                co_occurrence[i][j] = len(set(network[opening1]).intersection(set(network[opening2])))
    return openings, co_occurrence

def filter_relatedness_network(co_occurrence, alpha=0.05):
    z_scores = zscore(co_occurrence, axis=None)
    filtered_network = (z_scores > zscore(np.random.random(co_occurrence.shape))).astype(int)
    return filtered_network

def recommend_openings(user_openings, openings, co_occurrence, threshold=0.1):
    recommendations = defaultdict(float)
    explanations = defaultdict(list)
    for user_opening in user_openings:
        if user_opening in openings:
            idx = openings.index(user_opening)
            related_openings = co_occurrence[idx]
            for i, score in enumerate(related_openings):
                if score > threshold:
                    recommendations[openings[i]] += score
                    explanations[openings[i]].append(f"Related to your opening: {user_opening} (score: {score})")
    sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
    detailed_recommendations = [(opening, score, explanations[opening]) for opening, score in sorted_recommendations]
    return detailed_recommendations

# Example usage
if __name__ == "__main__":
    from extract_pgn import extract_pgn, extract_openings

    file_path = 'data/limited_lichess_games.pgn'  # Path to the extracted PGN file with limited games
    games = extract_pgn(file_path)
    openings = extract_openings(games)

    user_openings = ["Sicilian Defense", "French Defense"]
    network = build_player_opening_network(openings)
    openings, co_occurrence = project_opening_network(network)
    filtered_co_occurrence = filter_relatedness_network(co_occurrence)
    recommendations = recommend_openings(user_openings, openings, filtered_co_occurrence)

    for rec in recommendations:
        print(f"Opening: {rec[0]}, Score: {rec[1]}, Explanations: {rec[2]}")
