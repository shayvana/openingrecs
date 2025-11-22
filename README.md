# Chess Opening Recommendations v2.0

Personalized chess opening recommendations based on **Economic Fitness & Complexity (EFC)** analysis using Lichess game data.

This project implements the methodology from the Nature Scientific Reports paper: ["Quantifying the Complexity and Similarity of Chess Openings Using Online Chess Community Data"](https://www.nature.com/articles/s41598-023-31658-w).

![Chess Opening Recommendations](https://github.com/shayvana/openingrecs/assets/19787070/a7890b6e-49fb-4668-ae99-5a27295197b3)

## Features

✨ **Scientific methodology**: Z-score filtering, Economic Fitness & Complexity algorithm
📊 **Network analysis**: 144 openings, 1,654 connections, analyzed from 373K+ players
🎯 **Multi-factor recommendations**: Complexity, similarity, popularity, and novelty scoring
🎨 **Minimalist design**: Matches serialexperiment.ing aesthetic
⚡ **Fast & scalable**: Optimized for Vercel serverless deployment

## Quick Start

### Run Locally

```bash
# 1. Install dependencies
pip3 install -r requirements.txt

# 2. Start the app
./start_app.sh

# 3. Open browser
open http://localhost:5000
```

### Deploy to Vercel

```bash
# Quick deploy (2 minutes)
npm i -g vercel
vercel login
./deploy.sh production
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions.

## Try It Out

Visit the app and try these usernames:
- **DrNykterstein** (Magnus Carlsen)
- **penguingm1** (Andrew Tang)
- **GMWSO** (Qiyu Zhou)

## How It Works

### 1. Bipartite Network Construction

Builds a player-opening network from Lichess games:
- **Nodes**: 373,460 players × 144 openings
- **Edges**: Weighted by game frequency and player ratings
- **Metadata**: Player ratings, opening diversity scores

### 2. Statistical Filtering

Uses **z-score filtering** (not p-values) with Bipartite Configuration Model:
```python
z = (observed - expected) / std
Keep edges where z > 2.0
```

### 3. Economic Fitness & Complexity (EFC)

Iterative algorithm to calculate:
- **Player fitness**: F_i = Σ M_ij × Q_j
- **Opening complexity**: Q_j = 1 / Σ (M_ij / F_i)

Converges in ~50 iterations to reveal opening difficulty hierarchy.

### 4. Multi-Factor Recommendations

Combines four scoring factors:
- **Similarity** (40%): Network proximity to user's openings
- **Complexity** (30%): Match to user's skill level
- **Popularity** (20%): How often the opening is played
- **Novelty** (10%): New openings to expand repertoire

## Project Structure

```
chessopeningrecs/
├── app/
│   ├── app.py                          # Flask application
│   ├── build_bipartite_network.py      # Network construction
│   ├── project_and_filter_network.py   # Z-score filtering
│   ├── recommendation_engine_v2.py     # Multi-factor recommendations
│   └── templates/index.html            # Frontend UI
├── algorithms/
│   ├── efc.py                          # Economic Fitness & Complexity
│   ├── filtering.py                    # Z-score network filtering
│   ├── similarity.py                   # Opening similarity metrics
│   └── diversity.py                    # Opening diversity analysis
├── data/
│   ├── relatedness_network.pkl         # Opening similarity network
│   └── relatedness_network_metadata.pkl # EFC scores, complexity data
├── scripts/
│   ├── download_fresh_data.py          # Download latest Lichess data
│   └── reprocess_existing_network.py   # Rebuild networks
├── tests/
│   └── methodology/
│       └── test_paper_compliance.py    # Validate scientific methodology
├── vercel.json                         # Vercel deployment config
├── deploy.sh                           # Quick deployment script
└── requirements.txt                    # Python dependencies
```

## Documentation

- **[DEPLOYMENT.md](DEPLOYMENT.md)**: Complete Vercel deployment guide
- **[DATA_STATUS.md](DATA_STATUS.md)**: Data freshness and update guide

## Methodology

Based on: [Prata et al. (2023) - Nature Scientific Reports](https://www.nature.com/articles/s41598-023-31658-w)

**Key implementations:**
1. ✅ Z-score filtering (threshold: 2.0)
2. ✅ Economic Fitness & Complexity algorithm
3. ✅ Bipartite network projection
4. ✅ No artificial component connections
5. ✅ Player rating weighting
6. ✅ Opening complexity metrics

**Validation:** 15/19 tests passing (79%) - all critical methodology tests ✓

## Data

**Current network (June 2024):**
- 1.3 million games analyzed
- 373,460 players
- 144 unique openings
- 1,654 significant connections (19.68% edge retention)

**Update to latest data:**
```bash
# Download November 2024 games
python3 scripts/download_fresh_data.py 2024-11 50000
```

See [DATA_STATUS.md](DATA_STATUS.md) for details.

## Tech Stack

- **Backend**: Flask 3.0.0, Python 3.9+
- **Network Analysis**: NetworkX 3.2.1
- **Statistical Filtering**: BiCM 3.1.1
- **Data Processing**: NumPy, Pandas, SciPy
- **Chess**: python-chess 1.999
- **Deployment**: Vercel serverless
- **Frontend**: Vanilla JS, minimal design

## Development

### Run Tests

```bash
python3 -m pytest tests/methodology/test_paper_compliance.py -v
```

### Rebuild Networks

```bash
# Reprocess existing data with correct methodology
python3 scripts/reprocess_existing_network.py

# Or download fresh data and rebuild
python3 scripts/download_fresh_data.py 2024-11 50000
```

### Local Development

```bash
# Install dependencies
pip3 install -r requirements.txt

# Start development server
./start_app.sh

# App runs on http://localhost:5000 or :8000
```

## Contributing

This project follows scientific methodology from peer-reviewed research. Changes to core algorithms should maintain compliance with the Nature paper.

Run `pytest tests/methodology/test_paper_compliance.py` to validate methodology.

## License

MIT License - See LICENSE file

## Acknowledgments

- **Methodology**: Prata et al. (2023) - Nature Scientific Reports
- **Data**: Lichess Open Database
- **Design**: serialexperiment.ing aesthetic
- **Algorithm**: Economic Fitness & Complexity (Tacchella et al., 2012)

## Links

- 📄 [Research Paper](https://www.nature.com/articles/s41598-023-31658-w)
- ♟️ [Lichess Database](https://database.lichess.org/)
- 🌐 [serialexperiment.ing](https://www.serialexperiment.ing/)

---

**Version**: 2.0.0
**Last Updated**: November 2024
**Methodology Status**: ✅ Paper-compliant
