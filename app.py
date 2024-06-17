import requests
import re
import json
import pickle
import networkx as nx
from flask import Flask, render_template, request, jsonify
from collections import defaultdict

app = Flask(__name__, template_folder='app/templates', static_folder='app/static')

def fetch_games(username, num_games=100):
    url = f'https://lichess.org/api/games/user/{username}'
    headers = {
        'Accept': 'application/x-ndjson'
    }
    params = {
        'max': num_games,
        'opening': True,
        'pgnInJson': True  # Ensures PGN data is in JSON format
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        return response.text
    else:
        return None

def extract_openings(pgn_data):
    openings = []
    for line in pgn_data.strip().split('\n'):
        if line.strip():
            game_data = json.loads(line)
            if 'opening' in game_data:
                openings.append(game_data['opening']['name'])
    return openings

def normalize_opening(opening):
    return opening.split(":")[0].strip()

def recommend_openings(user_openings, relatedness_network):
    recommendations = defaultdict(float)
    explanations = defaultdict(list)
    normalized_user_openings = [normalize_opening(o) for o in user_openings]

    print(f"Normalized user openings: {normalized_user_openings}")

    for user_opening in normalized_user_openings:
        if user_opening in relatedness_network:
            for neighbor, edge_attrs in relatedness_network[user_opening].items():
                weight = edge_attrs.get('weight', 1.0)
                if neighbor not in normalized_user_openings:
                    recommendations[neighbor] += weight
                    explanations[neighbor].append(f"{user_opening}")

    sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
    top_result_explanation = None
    if sorted_recommendations:
        top_result, top_score = sorted_recommendations[0]
        unique_explanations = list(dict.fromkeys(explanations[top_result]))
        top_result_explanation = (f"The top recommendation is {top_result} with a score of {top_score} because it is related to "
                                  f"the openings you use frequently, such as {', '.join(unique_explanations)}.")
    
    return sorted_recommendations[:10], top_result_explanation

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    username = request.form['username']
    print(f"Received username: {username}")
    pgn_data = fetch_games(username)
    if pgn_data:
        user_openings = extract_openings(pgn_data)
        print(f"Extracted openings: {user_openings}")
        normalized_user_openings = [normalize_opening(o) for o in user_openings]
        print(f"Normalized user openings: {normalized_user_openings}")

        try:
            with open('data/relatedness_network.pkl', 'rb') as f:
                relatedness_network = pickle.load(f)
            print(f"Relatedness network nodes: {list(relatedness_network.nodes())[:10]}")
        except Exception as e:
            print(f"Error loading relatedness network: {e}")
            return jsonify({"error": "Failed to load the relatedness network."})
        
        recommendations, top_result_explanation = recommend_openings(normalized_user_openings, relatedness_network)
        print(f"Recommendations: {recommendations}")
        print(f"Top result explanation: {top_result_explanation}")
        return jsonify({'recommendations': recommendations, 'top_result_explanation': top_result_explanation})
    else:
        print("Failed to fetch games for the user.")
        return jsonify({"error": "Could not fetch games for the given username."})

if __name__ == '__main__':
    app.run(debug=True, port=5001, host='0.0.0.0')
