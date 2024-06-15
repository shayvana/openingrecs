import re

def extract_pgn(file_path, max_games=100000):
    games = []
    game = []
    game_count = 0

    with open(file_path, 'r') as f:
        for line in f:
            if line.strip() == '' and game:
                games.append('\n'.join(game))
                game = []
                game_count += 1
                if game_count >= max_games:
                    break
            else:
                game.append(line.strip())
    return games

def extract_openings(games):
    openings = []
    for game in games:
        opening = re.search(r'\[Opening "([^"]+)"\]', game)
        if opening:
            openings.append(opening.group(1))
    return openings

if __name__ == "__main__":
    file_path = 'data/limited_lichess_games.pgn'
    games = extract_pgn(file_path)
    openings = extract_openings(games)
    with open('data/openings.txt', 'w') as f:
        for opening in openings:
            f.write(f"{opening}\n")
