# Data Status and Freshness Guide

## Current Data Status

### What We Have Now ⚠️

**Data Age: June 17, 2024** (5+ months old)

```
data/
├── bipartite_network.pkl        (June 17, 2024) ← OLD DATA
├── relatedness_network.pkl      (Nov 21, 2025) ← REPROCESSED WITH NEW METHODOLOGY
└── relatedness_network_metadata.pkl
```

**What This Means:**
- ✅ **Methodology**: Correct (z-scores, EFC, no artificial connections)
- ⚠️ **Data**: From June 2024 games
- 📊 **Stats**: 373,460 players, 144 openings, 1.3M games

### What We Did During Migration

We **reprocessed the existing June 2024 network** with:
1. ✅ Z-score filtering (instead of p-values)
2. ✅ EFC algorithm (newly implemented)
3. ✅ No artificial component connections
4. ✅ Proper edge weighting

**Result**: Correct methodology, but old data.

## Getting Fresh Data

### Option 1: Quick Test (10,000 games) - Recommended First

**Time**: ~5-10 minutes
**Size**: ~50 MB

```bash
cd /Users/shayvana/Library/Mobile\ Documents/com~apple~CloudDocs/Documents/Code/Projects/chessopeningrecs

# Install zstandard if needed
pip3 install zstandard

# Download November 2024 data (10K games)
python3 scripts/download_fresh_data.py 2024-11 10000

# This will create:
# - data/lichess_2024-11_10000games.pgn
# - data/bipartite_network_fresh.pkl
# - data/relatedness_network_fresh.pkl
```

### Option 2: Medium Dataset (50,000 games)

**Time**: ~30-45 minutes
**Size**: ~250 MB

```bash
python3 scripts/download_fresh_data.py 2024-11 50000 1800
#                                        ^       ^      ^
#                                      month  games  min_rating
```

### Option 3: Large Dataset (500,000 games)

**Time**: ~3-5 hours
**Size**: ~2-3 GB

```bash
python3 scripts/download_fresh_data.py 2024-11 500000 2000
```

### Option 4: Professional Dataset (1,000,000 games)

**Time**: ~6-10 hours
**Size**: ~5 GB

```bash
python3 scripts/download_fresh_data.py 2024-10 1000000 2200
```

## Available Months

Lichess database has data for every month:

- **2024-11** (November 2024) - Most recent!
- **2024-10** (October 2024)
- **2024-09** (September 2024)
- **2024-08** (August 2024)
- Earlier months available at: https://database.lichess.org/

## After Downloading Fresh Data

The script creates files with `_fresh` suffix. To use them:

```bash
cd /Users/shayvana/Library/Mobile\ Documents/com~apple~CloudDocs/Documents/Code/Projects/chessopeningrecs

# Backup old files
mv data/bipartite_network.pkl data/bipartite_network_june2024.pkl
mv data/relatedness_network.pkl data/relatedness_network_june2024.pkl

# Use fresh files
mv data/bipartite_network_fresh.pkl data/bipartite_network.pkl
mv data/relatedness_network_fresh.pkl data/relatedness_network.pkl
mv data/relatedness_network_fresh_metadata.pkl data/relatedness_network_metadata.pkl

# Restart Flask app
./start_app.sh
```

## Comparison: Old vs Fresh

### Current (June 2024)
- **Games**: ~1.3 million
- **Players**: 373,460
- **Openings**: 144
- **Age**: 5+ months old
- **Quality**: Good (diverse dataset)

### Fresh (November 2024) - After Download
- **Games**: Your choice (10K - 1M+)
- **Players**: Varies
- **Openings**: Similar (~140-150)
- **Age**: Current month
- **Quality**: Latest meta, recent trends

## Should You Update?

### Reasons to Keep Current Data ✅
- It works fine for testing
- Large dataset (1.3M games)
- Good coverage of openings
- Methodology is already fixed

### Reasons to Get Fresh Data ⚠️
- Want latest opening trends
- Current meta has changed
- Need recent player data
- Testing with current games
- Production deployment

## Recommended Workflow

### For Testing/Development
```bash
# 1. Test with current data first
./start_app.sh

# 2. If it works well, optionally upgrade later
python3 scripts/download_fresh_data.py 2024-11 10000
```

### For Production
```bash
# 1. Download medium dataset
python3 scripts/download_fresh_data.py 2024-11 50000 1800

# 2. Test the fresh network
python3 app/app.py

# 3. If satisfied, replace old files
mv data/bipartite_network_fresh.pkl data/bipartite_network.pkl
mv data/relatedness_network_fresh.pkl data/relatedness_network.pkl

# 4. Deploy
```

## Script Options Explained

```bash
python3 scripts/download_fresh_data.py <month> <games> <min_rating>
#                                        ^       ^       ^
#                                        |       |       |
#    Format: YYYY-MM (e.g., 2024-11) ───┘       |       |
#    Number of games to extract (10000-1000000) ┘       |
#    Minimum player rating (1500-2500) ─────────────────┘
```

**min_rating recommendations:**
- `1500` - Include all serious players (more data)
- `1800` - Intermediate and above (balanced)
- `2000` - Strong players only (quality over quantity)
- `2200` - Master level (very high quality)

## Troubleshooting

### Script fails with "zstandard not found"
```bash
pip3 install zstandard
```

### Download is slow
- Normal! Lichess database files are large (10+ GB compressed)
- The script streams and stops after extracting your requested games
- Expect ~1-2 minutes per 10,000 games

### Not enough disk space
- Check space: `df -h`
- Each 10K games ≈ 50 MB
- Each 100K games ≈ 500 MB
- Each 1M games ≈ 5 GB

### Want to cancel mid-download
- Press `Ctrl+C`
- Partial PGN file will be saved
- You can process it anyway (fewer games than requested)

## Quick Reference

| Purpose | Command | Time | Size |
|---------|---------|------|------|
| **Quick test** | `python3 scripts/download_fresh_data.py 2024-11 10000` | ~5 min | ~50 MB |
| **Development** | `python3 scripts/download_fresh_data.py 2024-11 50000 1800` | ~30 min | ~250 MB |
| **Production** | `python3 scripts/download_fresh_data.py 2024-11 500000 2000` | ~3 hours | ~2.5 GB |
| **Research** | `python3 scripts/download_fresh_data.py 2024-10 1000000 2200` | ~8 hours | ~5 GB |

## Summary

**Current Status:**
- ⚠️ Using June 2024 data (old)
- ✅ Methodology is correct (just fixed!)
- ✅ App works fine for testing

**Next Step (Optional but Recommended):**
```bash
# Get fresh November 2024 data
pip3 install zstandard
python3 scripts/download_fresh_data.py 2024-11 10000
```

**Then replace old files and restart Flask app.**
