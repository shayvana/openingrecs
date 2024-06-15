import requests
import zstandard as zstd

def download_and_extract_pgn(url, output_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    wrote = 0

    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            for data in response.iter_content(block_size):
                wrote += len(data)
                f.write(data)
                print(f"\rProgress: {wrote * 100 / total_size:.2f}%", end="")
        print("\nDownload complete.")
        extract_zst(output_path)
    else:
        print("Failed to download the file.")

def extract_zst(file_path):
    dctx = zstd.ZstdDecompressor()
    output_file = file_path.replace('.zst', '')
    with open(file_path, 'rb') as compressed_file:
        with open(output_file, 'wb') as decompressed_file:
            decompressed_file.write(dctx.decompress(compressed_file.read()))
    print("Extraction complete.")

def extract_limited_games(input_path, output_path, max_games=1000):
    dctx = zstd.ZstdDecompressor()
    with open(input_path, 'rb') as compressed_file:
        with open(output_path, 'w') as decompressed_file:
            stream_reader = dctx.stream_reader(compressed_file)
            buffer = ""
            game_count = 0

            while True:
                chunk = stream_reader.read(16384)
                if not chunk:
                    break
                buffer += chunk.decode('utf-8')
                games = buffer.split('\n\n\n')

                for game in games[:-1]:  # Process all complete games
                    decompressed_file.write(game + '\n\n\n')
                    game_count += 1
                    if game_count >= max_games:
                        print(f"Extracted {game_count} games.")
                        return

                buffer = games[-1]  # Keep the incomplete game part for the next chunk

            # Write the remaining incomplete game if any
            if buffer.strip():
                decompressed_file.write(buffer + '\n\n\n')

    print(f"Extracted {game_count} games.")

# URL for April 2024 PGN file
url = 'https://database.lichess.org/standard/lichess_db_standard_rated_2024-04.pgn.zst'
output_path = 'data/lichess_db_standard_rated_2024-04.pgn.zst'

# pgn_path = 'data/limited_lichess_games.pgn'
# extract_limited_games(pgn_path, output_path)

# download_and_extract_pgn(url, output_path)
# Check the extracted PGN file content
file_path = 'data/lichess_db_standard_rated_2024-04.pgn'
with open(file_path, 'r') as f:
    print(f.read(1000))  # Print the first 1000 characters to verify the content

