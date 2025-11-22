"""
Download fresh data from Lichess and rebuild networks.

This script:
1. Downloads recent PGN data from Lichess database
2. Builds bipartite network from scratch
3. Applies z-score filtering
4. Calculates EFC metrics

Usage:
    python3 scripts/download_fresh_data.py [month] [num_games]

Examples:
    python3 scripts/download_fresh_data.py 2024-11 10000
    python3 scripts/download_fresh_data.py 2024-10 50000
"""

import requests
import sys
import os
import zstandard as zstd
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.build_bipartite_network import build_bipartite_network, save_network as save_bipartite
from app.project_and_filter_network import project_and_filter_network, save_network


def download_lichess_pgn(year_month, output_path, max_games=10000):
    """
    Download and extract limited games from Lichess database.

    Args:
        year_month: Format "YYYY-MM" (e.g., "2024-11")
        output_path: Path to save extracted PGN
        max_games: Maximum number of games to extract

    Returns:
        Path to extracted PGN file
    """
    url = f'https://database.lichess.org/standard/lichess_db_standard_rated_{year_month}.pgn.zst'

    print("=" * 70)
    print(f"Downloading Lichess Database: {year_month}")
    print("=" * 70)
    print(f"URL: {url}")
    print(f"Max games: {max_games}")
    print()

    # Download compressed file
    compressed_path = output_path + '.zst'

    print("Downloading compressed file (streaming)...")
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        # Get file size
        total_size = int(response.headers.get('content-length', 0))
        print(f"File size: {total_size / (1024**3):.2f} GB")

        if total_size > 5 * (1024**3):  # > 5 GB
            print("\n⚠️  WARNING: This file is very large!")
            print("We'll stream and extract only the first games to save time/space.")
            print()

        # Stream download and immediate extraction
        print("Streaming and extracting games...")
        dctx = zstd.ZstdDecompressor()

        with open(output_path, 'w') as out_file:
            stream_reader = dctx.stream_reader(response.raw)
            buffer = ""
            game_count = 0
            bytes_read = 0

            with tqdm(total=max_games, desc="Extracting games") as pbar:
                while game_count < max_games:
                    chunk = stream_reader.read(65536)  # 64KB chunks
                    if not chunk:
                        break

                    bytes_read += len(chunk)
                    buffer += chunk.decode('utf-8', errors='ignore')

                    # Split on game boundaries (empty lines)
                    games = buffer.split('\n\n\n')

                    # Process all complete games
                    for game in games[:-1]:
                        if game.strip():
                            out_file.write(game + '\n\n\n')
                            game_count += 1
                            pbar.update(1)

                            if game_count >= max_games:
                                break

                    # Keep incomplete game for next iteration
                    buffer = games[-1]

                    # Show progress
                    if bytes_read % (10 * 1024 * 1024) == 0:  # Every 10 MB
                        print(f"  Downloaded: {bytes_read / (1024**2):.1f} MB, "
                              f"Games: {game_count}/{max_games}")

        print(f"\n✅ Successfully extracted {game_count} games")
        print(f"Saved to: {output_path}")
        print(f"File size: {os.path.getsize(output_path) / (1024**2):.1f} MB")

        return output_path

    except requests.exceptions.RequestException as e:
        print(f"\n❌ Download failed: {e}")
        return None
    except Exception as e:
        print(f"\n❌ Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def rebuild_all(pgn_file, min_rating=1500):
    """
    Rebuild all networks from PGN file.

    Args:
        pgn_file: Path to PGN file
        min_rating: Minimum player rating to include

    Returns:
        True if successful
    """
    print("\n" + "=" * 70)
    print("Step 2: Building Bipartite Network")
    print("=" * 70)

    try:
        # Build bipartite network
        bipartite = build_bipartite_network(
            pgn_file=pgn_file,
            max_games=None,  # Process all games in file
            min_rating=min_rating
        )

        # Save bipartite network
        bipartite_path = 'data/bipartite_network_fresh.pkl'
        save_bipartite(bipartite, bipartite_path)

        print("\n" + "=" * 70)
        print("Step 3: Filtering and Calculating EFC")
        print("=" * 70)

        # Project and filter
        relatedness, metadata = project_and_filter_network(
            bipartite_file=bipartite_path,
            z_threshold=2.0,
            calculate_efc=True
        )

        # Save filtered network
        relatedness_path = 'data/relatedness_network_fresh.pkl'
        save_network(relatedness, metadata, relatedness_path)

        print("\n" + "=" * 70)
        print("✅ SUCCESS: All networks rebuilt!")
        print("=" * 70)

        print(f"\nFiles created:")
        print(f"  - {bipartite_path}")
        print(f"  - {relatedness_path}")
        print(f"  - {relatedness_path.replace('.pkl', '_metadata.pkl')}")

        print(f"\nTo use the new networks, replace the old ones:")
        print(f"  mv data/bipartite_network.pkl data/bipartite_network_old.pkl")
        print(f"  mv data/relatedness_network.pkl data/relatedness_network_old.pkl")
        print(f"  mv data/bipartite_network_fresh.pkl data/bipartite_network.pkl")
        print(f"  mv data/relatedness_network_fresh.pkl data/relatedness_network.pkl")

        return True

    except Exception as e:
        print(f"\n❌ Failed to rebuild networks: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/download_fresh_data.py <YYYY-MM> [num_games] [min_rating]")
        print()
        print("Examples:")
        print("  python3 scripts/download_fresh_data.py 2024-11 10000")
        print("  python3 scripts/download_fresh_data.py 2024-10 50000 1800")
        print()
        print("Recent months available:")
        print("  2024-11 (November 2024)")
        print("  2024-10 (October 2024)")
        print("  2024-09 (September 2024)")
        sys.exit(1)

    year_month = sys.argv[1]
    num_games = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
    min_rating = int(sys.argv[3]) if len(sys.argv) > 3 else 1500

    print("\n" + "=" * 70)
    print("Fresh Data Download and Network Rebuild")
    print("=" * 70)
    print(f"Month: {year_month}")
    print(f"Games to extract: {num_games:,}")
    print(f"Minimum rating: {min_rating}")
    print("=" * 70)
    print()

    # Download and extract
    pgn_file = f'data/lichess_{year_month}_{num_games}games.pgn'
    downloaded = download_lichess_pgn(year_month, pgn_file, num_games)

    if not downloaded:
        print("\n❌ Failed to download data")
        sys.exit(1)

    # Rebuild networks
    success = rebuild_all(downloaded, min_rating)

    if success:
        print("\n" + "=" * 70)
        print("🎉 All done! Fresh networks ready to use!")
        print("=" * 70)
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
