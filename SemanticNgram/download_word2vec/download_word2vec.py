import os
import requests
from tqdm import tqdm
import gzip
import shutil

def download_word2vec(destination_folder="model"):
    """Download and extract the Word2Vec Google News vectors."""
    url = "https://github.com/mmihaltz/word2vec-GoogleNews-vectors/raw/master/GoogleNews-vectors-negative300.bin.gz"
    compressed_file_path = os.path.join(destination_folder, "GoogleNews-vectors-negative300.bin.gz")
    extracted_file_path = os.path.join(destination_folder, "GoogleNews-vectors-negative300.bin")

    # Ensure the destination folder exists
    os.makedirs(destination_folder, exist_ok=True)

    # Download the file if it doesn't already exist
    if not os.path.exists(compressed_file_path):
        print(f"Downloading Word2Vec model from {url}...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        with open(compressed_file_path, "wb") as f, tqdm(
            desc="Downloading Word2Vec",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
                bar.update(len(chunk))
        print("Download complete.")

    # Extract the file if the extracted version doesn't already exist
    if not os.path.exists(extracted_file_path):
        print(f"Extracting {compressed_file_path}...")
        with gzip.open(compressed_file_path, "rb") as f_in:
            with open(extracted_file_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        print("Extraction complete.")

    print(f"Word2Vec model ready at: {extracted_file_path}")
    return extracted_file_path

if __name__ == "__main__":
    # Run this script to download and extract the model
    download_word2vec()
