"""
Flask web application for chess opening recommendations.

Updated to use corrected methodology from Nature paper.
"""

import requests
import json
import pickle
import networkx as nx
from flask import Flask, render_template, request, jsonify
import os
import logging
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import new recommendation engine
from app.recommendation_engine_v2 import RecommendationEngine

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates', static_folder='static')

# Global variable to cache the network
_relatedness_network = None
_recommendation_engine = None


def get_recommendation_engine():
    """Get or create recommendation engine (cached)."""
    global _relatedness_network, _recommendation_engine

    if _recommendation_engine is None:
        try:
            # Try new filename first
            network_path = 'data/relatedness_network.pkl'
            if not os.path.exists(network_path):
                # Fall back to old filename if it exists
                logger.warning(f"Network not found at {network_path}")
                return None

            logger.info(f"Loading relatedness network from {network_path}")
            with open(network_path, 'rb') as f:
                _relatedness_network = pickle.load(f)

            logger.info(f"Loaded network: {_relatedness_network.number_of_nodes()} nodes, "
                       f"{_relatedness_network.number_of_edges()} edges")

            # Create recommendation engine
            _recommendation_engine = RecommendationEngine(
                _relatedness_network,
                complexity_weight=0.3,
                similarity_weight=0.4,
                popularity_weight=0.2,
                novelty_weight=0.1
            )

            logger.info("Recommendation engine initialized")

        except Exception as e:
            logger.error(f"Error loading network or creating engine: {e}", exc_info=True)
            return None

    return _recommendation_engine


def fetch_games(username, num_games=100):
    """
    Fetch games for a user from Lichess API.

    Args:
        username: Lichess username
        num_games: Maximum number of games to fetch

    Returns:
        Response text or None on error
    """
    url = f'https://lichess.org/api/games/user/{username}'
    headers = {
        'Accept': 'application/x-ndjson'
    }
    params = {
        'max': num_games,
        'opening': True,
        'pgnInJson': True,
        'rated': True  # Prefer rated games to get ratings
    }

    try:
        logger.info(f"Fetching games for user: {username}")
        response = requests.get(url, headers=headers, params=params, timeout=30)

        if response.status_code == 200:
            logger.info(f"Successfully fetched games for {username}")
            return response.text
        elif response.status_code == 404:
            logger.warning(f"User not found: {username}")
            return None
        else:
            logger.error(f"API error: {response.status_code}")
            return None

    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        return None


def extract_openings_and_rating(pgn_data, username):
    """
    Extract openings and user rating from PGN data.

    Args:
        pgn_data: NDJSON response from Lichess API
        username: Username to extract rating for

    Returns:
        Tuple of (openings_list, average_rating)
    """
    openings = []
    ratings = []

    for line in pgn_data.strip().split('\n'):
        if not line.strip():
            continue

        try:
            game_data = json.loads(line)

            # Extract opening
            if 'opening' in game_data and 'name' in game_data['opening']:
                openings.append(game_data['opening']['name'])

            # Extract rating for this user
            if 'players' in game_data:
                for color in ['white', 'black']:
                    if (color in game_data['players'] and
                        'user' in game_data['players'][color] and
                        game_data['players'][color]['user'].get('name', '').lower() == username.lower()):

                        if 'rating' in game_data['players'][color]:
                            ratings.append(game_data['players'][color]['rating'])
                        break

        except json.JSONDecodeError:
            logger.warning(f"Failed to parse game data: {line[:100]}")
            continue

    avg_rating = sum(ratings) / len(ratings) if ratings else None

    logger.info(f"Extracted {len(openings)} openings, average rating: {avg_rating}")

    return openings, avg_rating


@app.route('/')
def index():
    """Render homepage."""
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Generate opening recommendations for a user.

    Expects JSON with 'username' field.
    Returns JSON with recommendations and explanations.
    """
    try:
        # Get username from request
        username = request.form.get('username') or request.json.get('username')

        if not username:
            return jsonify({"error": "Username is required"}), 400

        username = username.strip()
        logger.info(f"Processing recommendation request for: {username}")

        # Get recommendation engine
        engine = get_recommendation_engine()
        if engine is None:
            logger.error("Recommendation engine not available")
            return jsonify({
                "error": "Recommendation system not available. Please ensure the network has been built."
            }), 500

        # Fetch user games
        pgn_data = fetch_games(username, num_games=100)

        if not pgn_data:
            logger.warning(f"No games found for user: {username}")
            return jsonify({
                "error": f"Could not fetch games for user '{username}'. Please check the username is correct."
            }), 404

        # Extract openings and rating
        user_openings, user_rating = extract_openings_and_rating(pgn_data, username)

        if not user_openings:
            logger.warning(f"No openings found for user: {username}")
            return jsonify({
                "error": "No games with opening information found. Please play more games!"
            }), 400

        logger.info(f"User openings: {user_openings[:5]}... (total: {len(user_openings)})")
        logger.info(f"User rating: {user_rating}")

        # Generate recommendations using new engine
        recommendations = engine.recommend(
            user_openings=user_openings,
            user_rating=user_rating,
            top_k=10,
            include_explanations=True
        )

        if not recommendations:
            logger.warning("No recommendations generated")
            return jsonify({
                "error": "Could not generate recommendations. Your openings may not be in the database."
            }), 400

        # Format recommendations for response
        formatted_recommendations = []
        for opening, score, explanation in recommendations:
            # Get complexity if available
            complexity = None
            if opening in _relatedness_network.nodes:
                complexity = _relatedness_network.nodes[opening].get('complexity')

            formatted_recommendations.append({
                'opening': opening,
                'score': round(score, 3),
                'explanation': explanation,
                'complexity': round(complexity, 3) if complexity else None
            })

        # Get top explanation
        top_explanation = recommendations[0][2] if recommendations else "No recommendations available"

        # Additional stats
        response = {
            'recommendations': formatted_recommendations,
            'top_result_explanation': top_explanation,
            'user_stats': {
                'games_analyzed': len(user_openings),
                'unique_openings': len(set(user_openings)),
                'rating': user_rating
            }
        }

        logger.info(f"Successfully generated {len(recommendations)} recommendations")

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error processing recommendation: {e}", exc_info=True)
        return jsonify({
            "error": "An internal error occurred. Please try again later."
        }), 500


@app.route('/health')
def health():
    """Health check endpoint."""
    engine = get_recommendation_engine()

    if engine is None:
        return jsonify({"status": "unhealthy", "message": "Network not loaded"}), 503

    return jsonify({
        "status": "healthy",
        "network_nodes": _relatedness_network.number_of_nodes() if _relatedness_network else 0,
        "network_edges": _relatedness_network.number_of_edges() if _relatedness_network else 0
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'

    logger.info(f"Starting Flask app on port {port}, debug={debug}")
    app.run(debug=debug, port=port, host='0.0.0.0')
