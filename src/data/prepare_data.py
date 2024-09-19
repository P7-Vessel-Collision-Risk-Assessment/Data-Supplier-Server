import os
import pandas as pd
import numpy as np
from pandas.io.parsers import TextFileReader
import argparse
from tqdm import tqdm
import subprocess

CHUNK_SIZE = 10**5


def count_lines(path: str) -> int:
    return int(subprocess.check_output(["wc", "-l", path]).split()[0])


def load_data(path: str) -> TextFileReader:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path, chunksize=CHUNK_SIZE)


def cog_to_xy(cog: float) -> tuple[float, float]:
    x = np.sin(np.radians(cog))
    y = np.cos(np.radians(cog))
    return x, y


def group_data(data: TextFileReader, dir: str):
    os.makedirs(dir, exist_ok=True)

    total_chunks = count_lines(args.path) // CHUNK_SIZE

    pbar = tqdm(total=total_chunks, desc="Processing chunks")

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
        chunk.columns = chunk.columns.str.lower()

        grouped = chunk.groupby("mmsi")

        # Convert COG to x, y
        chunk["cog_x"], chunk["cog_y"] = zip(*chunk["cog"].map(cog_to_xy))

        for name, group in grouped:
            group = group.drop_duplicates()
            if os.path.exists(f"{dir}/{name}.feather"):
                combined_data = pd.read_feather(f"{dir}/{name}.feather")
                combined_data = pd.concat([combined_data, group])
                combined_data.to_feather(f"{dir}/{name}.feather")
                del combined_data
            else:
                group.to_feather(f"{dir}/{name}.feather")

        pbar.update(1)

    pbar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to the AIS data CSV file")
    parser.add_argument(
        "--output-dir", default="data/mmsi2/", help="Output directory for grouped data"
    )
    args = parser.parse_args()

    # Load data and process with progress bar
    data = load_data(args.path)
    group_data(data, args.output_dir)

    print(f"Data '{args.path}' grouped by MMSI and saved to '{args.output_dir}'")
