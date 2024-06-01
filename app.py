import requests
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def fetch_user_games(username, max_games=200):
    url = f"https://lichess.org/api/games/user/{username}?max={max_games}&opening=true"
    headers = {"Accept": "application/x-ndjson"}
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"Error fetching games: {response.status_code}")
        return []

    games = []
    for line in response.iter_lines():
        if line:
            try:
                games.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Error decoding JSON: {line}")
    
    return games

def parse_games(games, username):
    data = {
        "username": [],
        "rating": [],
        "opening": [],
        "result": []
    }
    
    for game in games:
        try:
            user_color = 'white' if game['players']['white']['user']['name'].lower() == username.lower() else 'black'
            opponent_color = 'black' if user_color == 'white' else 'white'
            
            # Check if all necessary keys are present
            if ('user' in game['players'][user_color] and
                'name' in game['players'][user_color]['user'] and
                'rating' in game['players'][user_color] and
                'opening' in game and
                'name' in game['opening']):
                
                data["username"].append(game['players'][user_color]['user']['name'])
                data["rating"].append(game['players'][user_color]['rating'])
                data["opening"].append(game['opening']['name'])
                data["result"].append(game['winner'] if 'winner' in game else 'draw')
            else:
                print(f"Skipping game due to missing data: {game}")
                
        except KeyError as e:
            print(f"Missing key in game data: {e}")
    
    return pd.DataFrame(data)

def extract_features(df, username):
    df['win'] = df['result'].apply(lambda x: 1 if (x == 'white' and df['username'][0].lower() == username.lower()) or (x == 'black' and df['username'][0].lower() != username.lower()) else 0)
    df['draw'] = df['result'].apply(lambda x: 1 if x == 'draw' else 0)
    
    feature_df = df.groupby(['username', 'opening']).agg(
        total_games=pd.NamedAgg(column='result', aggfunc='count'),
        win_rate=pd.NamedAgg(column='win', aggfunc='mean'),
        draw_rate=pd.NamedAgg(column='draw', aggfunc='mean')
    ).reset_index()
    
    return feature_df

def calculate_similarity(feature_df):
    pivot_df = feature_df.pivot(index='username', columns='opening', values='win_rate').fillna(0)
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(pivot_df)
    
    similarity_matrix = cosine_similarity(normalized_features)
    
    return pivot_df, similarity_matrix

def find_similar_players(username, pivot_df, similarity_matrix, threshold=0.9):
    user_index = pivot_df.index.get_loc(username)
    similar_users = [
        pivot_df.index[i] for i, similarity in enumerate(similarity_matrix[user_index])
        if i != user_index and similarity >= threshold
    ]
    
    return similar_users

def recommend_openings(username, pivot_df, similar_users):
    user_openings = pivot_df.loc[username]
    similar_openings = pivot_df.loc[similar_users].mean()
    
    recommendations = similar_openings[user_openings == 0].sort_values(ascending=False)
    
    return recommendations.head(5)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    username = data['username']
    
    games = fetch_user_games(username)
    if not games:
        return jsonify({"error": "No games found or error fetching games"}), 400
    
    df = parse_games(games, username)
    if df.empty:
        return jsonify({"error": "No valid game data found"}), 400

    feature_df = extract_features(df, username)
    pivot_df, similarity_matrix = calculate_similarity(feature_df)
    
    similar_users = find_similar_players(username, pivot_df, similarity_matrix)
    recommendations = recommend_openings(username, pivot_df, similar_users)
    
    return jsonify(recommendations.index.tolist())

if __name__ == '__main__':
    app.run(debug=True)
