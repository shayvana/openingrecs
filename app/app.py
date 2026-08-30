import json
import logging
import os
import sys

import requests
from flask import Flask, jsonify, render_template, request

# Allow both `python3 app/app.py` (dev) and Vercel's Python builder to find modules.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from network import ChessNetwork
from recommendation_engine_v2 import RecommendationEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s %(message)s',
)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates', static_folder='static')

_network: ChessNetwork | None = None
_network_meta: dict = {}
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
    """Load the network from JSON and build the recommendation engine on first call."""
    global _network, _network_meta, _engine
    if _engine is not None:
        return _engine

    try:
        network_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'network.json'
        )
        if not os.path.exists(network_path):
            logger.error("Network file not found: %s", network_path)
            return None

        _network, _network_meta = ChessNetwork.from_json_with_meta(network_path)
        _network.remove_node('?')

        logger.info(
            "Network loaded: %d openings, %d edges (data: %s)",
            _network.number_of_nodes(),
            _network.number_of_edges(),
            _network_meta.get('month', 'unknown'),
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


def _parse_lichess(ndjson: str, username: str) -> tuple[list[str], list[str], float | None]:
    white_openings, black_openings, ratings = [], [], []
    username_lower = username.lower()
    for line in ndjson.strip().splitlines():
        if not line:
            continue
        try:
            game = json.loads(line)
            opening_name = game.get('opening', {}).get('name')
            for color in ('white', 'black'):
                player = game.get('players', {}).get(color, {})
                if player.get('user', {}).get('name', '').lower() == username_lower:
                    if 'rating' in player:
                        ratings.append(player['rating'])
                    if opening_name:
                        (white_openings if color == 'white' else black_openings).append(opening_name)
                    break
        except json.JSONDecodeError:
            continue
    avg_rating = sum(ratings) / len(ratings) if ratings else None
    return white_openings, black_openings, avg_rating


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


def _parse_chesscom(games: list, username: str) -> tuple[list[str], list[str], float | None]:
    white_openings, black_openings, ratings = [], [], []
    username_lower = username.lower()
    for game in games:
        try:
            eco_url = game.get('eco', '')
            if '/openings/' not in eco_url:
                continue
            slug = eco_url.split('/openings/')[-1].split('?')[0]
            name = slug.replace('-', ' ').strip()
            if not name or name.lower() in ('undefined', 'unknown'):
                continue
            for color in ('white', 'black'):
                side = game.get(color, {})
                if side.get('username', '').lower() == username_lower:
                    if 'rating' in side:
                        ratings.append(side['rating'])
                    (white_openings if color == 'white' else black_openings).append(name)
                    break
        except (KeyError, AttributeError):
            continue
    avg_rating = sum(ratings) / len(ratings) if ratings else None
    return white_openings, black_openings, avg_rating


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
    body = request.get_json(silent=True) or {}
    username = (form.get('username') or body.get('username', '')).strip()
    platform = (form.get('platform') or body.get('platform', 'auto')).strip().lower()

    if not username:
        return jsonify({'error': 'Username is required'}), 400

    engine = get_engine()
    if engine is None:
        return jsonify({'error': 'Recommendation system unavailable — network data missing'}), 500

    white_openings: list[str] = []
    black_openings: list[str] = []
    user_rating: float | None = None
    platform_used: str | None = None

    if platform == 'lichess':
        raw = _fetch_lichess(username)
        if raw:
            white_openings, black_openings, user_rating = _parse_lichess(raw, username)
            platform_used = 'Lichess'
    elif platform in ('chesscom', 'chess.com'):
        raw = _fetch_chesscom(username)
        if raw:
            white_openings, black_openings, user_rating = _parse_chesscom(raw, username)
            platform_used = 'Chess.com'
    else:
        raw = _fetch_lichess(username)
        if raw:
            white_openings, black_openings, user_rating = _parse_lichess(raw, username)
            platform_used = 'Lichess'
        else:
            raw = _fetch_chesscom(username)
            if raw:
                white_openings, black_openings, user_rating = _parse_chesscom(raw, username)
                platform_used = 'Chess.com'

    all_openings = white_openings + black_openings
    if not all_openings:
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

    white_recs = engine.recommend(white_openings, user_rating, top_k=5) if white_openings else []
    black_recs = engine.recommend(black_openings, user_rating, top_k=5) if black_openings else []

    if not white_recs and not black_recs:
        return jsonify({'error': "No recommendations — your openings may not be in the database."}), 400

    user_rarity = engine.estimate_rarity(all_openings, user_rating)

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
        'white': {
            'recommendations': [format_rec(*r) for r in white_recs],
            'games': len(white_openings),
            'unique_openings': len(set(white_openings)),
        },
        'black': {
            'recommendations': [format_rec(*r) for r in black_recs],
            'games': len(black_openings),
            'unique_openings': len(set(black_openings)),
        },
        'user_stats': {
            'total_games': len(all_openings),
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
        'data_month': _network_meta.get('month'),
        'built_at': _network_meta.get('built_at'),
        'games_processed': _network_meta.get('games_processed'),
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(debug=debug, port=port, host='0.0.0.0')
