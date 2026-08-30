import json
import logging
import os
import pickle
import sys

import requests
from flask import Flask, jsonify, render_template, request

# Add this file's directory to sys.path so recommendation_engine_v2 is importable
# both when run directly (python3 app/app.py) and via Vercel's Python builder.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recommendation_engine_v2 import RecommendationEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s %(message)s',
)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates', static_folder='static')

_network = None
_engine: RecommendationEngine | None = None


# ------------------------------------------------------------------
# Label helpers
# ------------------------------------------------------------------

def _rarity_label(rarity: float) -> str:
    """Human-readable label for an EFC rarity score."""
    if rarity < 0.1:
        return "Very Popular"
    if rarity < 1.0:
        return "Popular"
    if rarity < 5.0:
        return "Niche"
    return "Obscure"


def _repertoire_style(median_rarity: float) -> str:
    """Describe how mainstream or niche a player's opening repertoire is."""
    if median_rarity < 0.1:
        return "Mainstream"
    if median_rarity < 1.0:
        return "Varied"
    if median_rarity < 5.0:
        return "Niche"
    return "Highly Specialized"


# ------------------------------------------------------------------
# Engine bootstrap
# ------------------------------------------------------------------

def get_engine() -> RecommendationEngine | None:
    """Load the network and build the recommendation engine on first call."""
    global _network, _engine
    if _engine is not None:
        return _engine

    try:
        network_path = 'data/relatedness_network_v2.pkl'
        if not os.path.exists(network_path):
            logger.error("Network file not found: %s", network_path)
            return None

        with open(network_path, 'rb') as f:
            _network = pickle.load(f)

        if '?' in _network:
            _network.remove_node('?')

        metadata_path = 'data/relatedness_network_v2_metadata.pkl'
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            rarity_scores = metadata.get('opening_complexity', {})
        else:
            logger.warning("Metadata file not found; rarity scores will default to 1.0")
            rarity_scores = {}

        for node in _network.nodes():
            _network.nodes[node]['complexity'] = rarity_scores.get(node, 1.0)

        logger.info(
            "Network loaded: %d openings, %d edges",
            _network.number_of_nodes(),
            _network.number_of_edges(),
        )
        _engine = RecommendationEngine(_network)
        return _engine

    except Exception:
        logger.exception("Failed to load network or build recommendation engine")
        return None


# ------------------------------------------------------------------
# Game fetching
# ------------------------------------------------------------------

def _fetch_lichess(username: str, num_games: int = 100) -> str | None:
    try:
        resp = requests.get(
            f'https://lichess.org/api/games/user/{username}',
            headers={'Accept': 'application/x-ndjson'},
            params={'max': num_games, 'opening': True, 'pgnInJson': True, 'rated': True},
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.text
        if resp.status_code != 404:
            logger.error("Lichess API returned %d for %s", resp.status_code, username)
        return None
    except requests.RequestException as e:
        logger.error("Lichess request failed: %s", e)
        return None


def _parse_lichess(ndjson: str, username: str) -> tuple[list[str], float | None]:
    openings, ratings = [], []
    username_lower = username.lower()
    for line in ndjson.strip().splitlines():
        if not line:
            continue
        try:
            game = json.loads(line)
            if 'opening' in game and 'name' in game['opening']:
                openings.append(game['opening']['name'])
            for color in ('white', 'black'):
                player = game.get('players', {}).get(color, {})
                if player.get('user', {}).get('name', '').lower() == username_lower:
                    if 'rating' in player:
                        ratings.append(player['rating'])
                    break
        except json.JSONDecodeError:
            continue
    avg_rating = sum(ratings) / len(ratings) if ratings else None
    return openings, avg_rating


def _fetch_chesscom(username: str, num_games: int = 100) -> list | None:
    headers = {'User-Agent': 'OpeningRecs/2.0 (contact@serialexperiment.ing)'}
    try:
        resp = requests.get(
            f'https://api.chess.com/pub/player/{username}/games/archives',
            headers=headers,
            timeout=30,
        )
        if resp.status_code == 404:
            return None
        if resp.status_code != 200:
            logger.error("Chess.com archives API returned %d", resp.status_code)
            return None

        archives = resp.json().get('archives', [])
        if not archives:
            return None

        games = []
        for url in reversed(archives):
            if len(games) >= num_games:
                break
            r = requests.get(url, headers=headers, timeout=30)
            if r.status_code == 200:
                games.extend(r.json().get('games', []))

        return games[:num_games] if games else None

    except requests.RequestException as e:
        logger.error("Chess.com request failed: %s", e)
        return None


def _parse_chesscom(games: list, username: str) -> tuple[list[str], float | None]:
    openings, ratings = [], []
    username_lower = username.lower()
    for game in games:
        try:
            eco_url = game.get('eco', '')
            if '/openings/' in eco_url:
                slug = eco_url.split('/openings/')[-1].split('?')[0]
                name = slug.replace('-', ' ').strip()
                if name and name.lower() not in ('undefined', 'unknown'):
                    openings.append(name)
            for color in ('white', 'black'):
                side = game.get(color, {})
                if side.get('username', '').lower() == username_lower:
                    if 'rating' in side:
                        ratings.append(side['rating'])
                    break
        except (KeyError, AttributeError):
            continue
    avg_rating = sum(ratings) / len(ratings) if ratings else None
    return openings, avg_rating


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/methodology')
def methodology():
    return render_template('methodology.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    form = request.form
    body = request.json or {}
    username = (form.get('username') or body.get('username', '')).strip()
    platform = (form.get('platform') or body.get('platform', 'auto')).strip().lower()

    if not username:
        return jsonify({'error': 'Username is required'}), 400

    engine = get_engine()
    if engine is None:
        return jsonify({'error': 'Recommendation system unavailable — network data missing'}), 500

    user_openings, user_rating, platform_used = None, None, None

    if platform == 'lichess':
        raw = _fetch_lichess(username)
        if raw:
            user_openings, user_rating = _parse_lichess(raw, username)
            platform_used = 'Lichess'
    elif platform in ('chesscom', 'chess.com'):
        raw = _fetch_chesscom(username)
        if raw:
            user_openings, user_rating = _parse_chesscom(raw, username)
            platform_used = 'Chess.com'
    else:
        raw = _fetch_lichess(username)
        if raw:
            user_openings, user_rating = _parse_lichess(raw, username)
            platform_used = 'Lichess'
        else:
            raw = _fetch_chesscom(username)
            if raw:
                user_openings, user_rating = _parse_chesscom(raw, username)
                platform_used = 'Chess.com'

    if not user_openings:
        if platform_used == 'Chess.com':
            return jsonify({
                'error': (
                    f"No opening data for Chess.com user '{username}'. "
                    "Chess.com only stores openings for recent games — try Lichess instead."
                )
            }), 400
        if platform in ('lichess', 'chesscom', 'chess.com'):
            return jsonify({
                'error': f"Could not fetch games for '{username}' on {platform}. Check the username."
            }), 404
        return jsonify({
            'error': f"Could not find '{username}' on Lichess or Chess.com. Check the username."
        }), 404

    recs = engine.recommend(user_openings, user_rating, top_k=10)
    if not recs:
        return jsonify({'error': "No recommendations — your openings may not be in the database."}), 400

    user_rarity = engine.estimate_rarity(user_openings, user_rating)

    def format_rec(opening, score, explanation, n_supporting):
        if opening not in _network.nodes:
            return {
                'opening': opening,
                'score': round(score, 3),
                'explanation': explanation,
                'rarity': None,
                'rarity_label': 'Unknown',
                'n_supporting': n_supporting,
                'related_openings': [],
                'num_connections': 0,
            }
        rarity = _network.nodes[opening].get('complexity', 0.5)
        neighbors = sorted(
            (
                (n, _network[opening][n].get('z_score', 0))
                for n in _network.neighbors(opening)
                if n and len(n) > 1
            ),
            key=lambda x: -x[1],
        )
        return {
            'opening': opening,
            'score': round(score, 3),
            'explanation': explanation,
            'rarity': round(rarity, 4),
            'rarity_label': _rarity_label(rarity),
            'n_supporting': n_supporting,
            'related_openings': [n for n, _ in neighbors[:3]],
            'num_connections': int(_network.degree(opening)),
        }

    return jsonify({
        'recommendations': [format_rec(*r) for r in recs],
        'top_result_explanation': recs[0][2],
        'platform': platform_used,
        'user_stats': {
            'games_analyzed': len(user_openings),
            'unique_openings': len(set(user_openings)),
            'rating': user_rating,
            'repertoire_style': _repertoire_style(user_rarity),
            'platform': platform_used,
        },
    })


@app.route('/health')
def health():
    engine = get_engine()
    if engine is None:
        return jsonify({'status': 'unhealthy', 'message': 'Network not loaded'}), 503
    return jsonify({
        'status': 'healthy',
        'network_nodes': _network.number_of_nodes(),
        'network_edges': _network.number_of_edges(),
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(debug=debug, port=port, host='0.0.0.0')
