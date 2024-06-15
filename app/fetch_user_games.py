import requests
import json
import pandas as pd

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
