import argparse
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from pandas.io.parsers import TextFileReader
from tqdm import tqdm

pd.options.mode.chained_assignment = None

CHUNK_SIZE = 10**6


def count_lines(path: str) -> int:
    return int(subprocess.check_output(["wc", "-l", path]).split()[0])


def cog_to_xy(cog: float) -> tuple[float, float]:
    x = np.sin(np.radians(cog))
    y = np.cos(np.radians(cog))
    return x, y


def load_data(path: str) -> TextFileReader:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path, chunksize=CHUNK_SIZE)


def process_group(name, group, dir):
    file_path = f"{dir}/{name}.feather"
    if os.path.exists(file_path):
        combined_data = pd.read_feather(file_path)
        combined_data = pd.concat([combined_data, group])
        combined_data.to_feather(file_path)
    else:
        group.to_feather(file_path)


def group_data(data: TextFileReader, dir: str):
    os.makedirs(dir, exist_ok=True)

    total_chunks = count_lines(args.path) // CHUNK_SIZE

    pbar = tqdm(total=total_chunks, desc="Processing chunks")

    with ProcessPoolExecutor() as executor:
        for chunk in data:
            chunk = chunk[
                [
                    "# Timestamp",
                    "Type of mobile",
                    "MMSI",
                    "Latitude",
                    "Longitude",
                    "SOG",
                    "COG",
                    "Heading",
                    "Width",
                    "Length",
                ]
            ].rename(columns={"# Timestamp": "Timestamp"})

            chunk = chunk[chunk["Type of mobile"] == "Class A"]

            chunk.columns = chunk.columns.str.lower()

            chunk["cog_x"], chunk["cog_y"] = zip(*chunk["cog"].map(cog_to_xy))

            # Group by MMSI (vessel identifier)
            grouped = chunk.groupby("mmsi")

            # Submit each group for parallel processing
            futures = [
                executor.submit(process_group, name, group, dir)
                for name, group in grouped
            ]

            for future in futures:
                future.result()

            pbar.update(1)

    pbar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to the AIS data CSV file")
    parser.add_argument(
        "--output-dir", default="data/mmsi/", help="Output directory for grouped data"
    )
    args = parser.parse_args()

    data = load_data(args.path)
    group_data(data, args.output_dir)

    print(f"Data '{args.path}' grouped by MMSI and saved to '{args.output_dir}'")
