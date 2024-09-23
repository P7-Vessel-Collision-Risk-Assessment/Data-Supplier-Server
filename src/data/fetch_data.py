"""
Scraper to scape AIS data:
    index site: https://web.ais.dk/aisdata/
    example link: https://web.ais.dk/aisdata/aisdk-2024-09-16.zip
"""

import requests
from bs4 import BeautifulSoup
import os
import zipfile
import io
from tqdm import tqdm
import argparse

BASE_URL = "https://web.ais.dk/aisdata/"
DATA_DIR = "data"


def download_data(url: str, dir: str):
    os.makedirs(dir, exist_ok=True)

    # Stream the download and show progress
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte

    progress_bar = tqdm(
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
        desc=f"Downloading {url.split('/')[-1]}",
    )

    # Write to a BytesIO buffer
    with io.BytesIO() as buffer:
        for data in response.iter_content(block_size):
            buffer.write(data)
            progress_bar.update(len(data))

        progress_bar.close()
        print(f"Downloaded {url.split('/')[-1]}")
        # Reset buffer position to start
        buffer.seek(0)
        print("Extracting data...")
        # Extract the zip file
        with zipfile.ZipFile(buffer) as z:
            z.extractall(dir)

        print("Data extracted")


def get_latest_data_url():
    response = requests.get(BASE_URL)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    links = soup.find_all("a")

    links = [BASE_URL + link["href"] for link in links if link["href"].endswith(".zip")]
    print(f"Found {len(links)} links")
    return links[-2]


if __name__ == "__main__":
    url = get_latest_data_url()
    download_data(url, DATA_DIR)
    print(f"Data downloaded to {DATA_DIR}")
