import argparse
import subprocess
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import polars as pl
import os
import numpy as np
from polars.io.csv import BatchedCsvReader

CHUNK_SIZE = 10**6

def count_lines(path: str) -> int:
    return int(subprocess.check_output(["wc", "-l", path]).split()[0])

def cog_to_xy(cog: float) -> tuple[float, float]:
    x = np.sin(np.radians(cog))
    y = np.cos(np.radians(cog))
    return x, y

def load_data(path: str) -> pl.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pl.read_csv_batched(path, batch_size=CHUNK_SIZE, has_header=True, n_threads=4)

def build_trajectories(data: pl.DataFrame):
    data = data.unique(subset=None, keep="first")

    # Convert timestamp column to datetime
    data = data.with_columns([
        pl.col("timestamp").str.strptime(pl.Datetime, fmt="%d/%m/%Y %H:%M:%S").alias("timestamp")
    ])

    cols = ['latitude', 'longitude', 'sog', 'cog_x', 'cog_y']

    # Drop rows with NaN in required columns
    data = data.drop_nulls(subset=cols)

    # Return if dataframe is empty after removing rows with NaN values
    if data.height == 0:
        return None

    # Resample at 1-minute intervals and interpolate
    data = data.sort("timestamp").group_by_dynamic(
        "timestamp", every="1m", closed="left"
    ).agg([pl.col(cols).mean().interpolate()])

    # scaler = MinMaxScaler(feature_range=(-1, 1))
    # normalized_cols = scaler.fit_transform(data.select(cols).to_numpy())

    # Convert normalized columns back to DataFrame and return
    # data = data.with_columns([
    #     pl.Series(cols[i], normalized_cols[:, i]) for i in range(len(cols))
    # ])

    return data.select(["latitude", "longitude", "sog", "cog_x", "cog_y"]).to_numpy()

def trajectory_to_segments(df: pl.DataFrame, segment_length=30):
    segments = []

    # Iterate over each segment and store if it is 30 rows long
    for start in range(0, len(df) - segment_length + 1, segment_length):
        segment = df.slice(start, segment_length)
        if segment.height == segment_length:
            segments.append(segment.select(["latitude", "longitude", "sog", "cog_x", "cog_y"]).to_numpy())

    if len(segments) == 0:
        return None

    return np.array(segments)

def group_data(data: BatchedCsvReader, dir: str):
    os.makedirs(dir, exist_ok=True)

    total_chunks = count_lines(args.path) // CHUNK_SIZE

    pbar = tqdm(total=total_chunks, desc="Processing chunks")

    for chunk in data.next_batches(total_chunks):
        chunk = chunk.select([
            "# Timestamp", "Type of mobile", "MMSI", "Latitude", "Longitude", "SOG", "COG", "Heading", "Width", "Length"
        ]).rename({"# Timestamp": "timestamp"})

        chunk.columns = [x.lower() for x in chunk.columns]

        chunk = chunk.filter(pl.col("type of mobile") == "Class A")

        # Convert column names to lowercase
        chunk = chunk.with_columns([pl.col(c).alias(c.lower()) for c in chunk.columns])

        # Apply cog_to_xy function to generate cog_x and cog_y
        chunk = chunk.with_columns([
            pl.col("cog").map_elements(lambda cog: cog_to_xy(cog)[0], return_dtype=pl.Float32).alias("cog_x"),
            pl.col("cog").map_elements(lambda cog: cog_to_xy(cog)[1], return_dtype=pl.Float32).alias("cog_y")
        ])

        chunk = chunk.drop("cog")

        # Group by MMSI (vessel identifier)
        grouped = chunk.group_by("mmsi")

        # Process each vessel's data
        for group in grouped:
            name = group[0]
            vessel_data = group[1]

            file_path = f"{dir}/{name}.feather"

            if os.path.exists(file_path):
                with open(file_path, mode="a") as combined_data:
                    vessel_data.write_ipc(combined_data)
            else:
                vessel_data.write_ipc(file_path)

        pbar.update(1)

    pbar.close()

def build_dataset(in_dir="data/mmsi", out_dir="data/trajectories"):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    for filename in os.listdir(in_dir):
        if filename.endswith('.feather'):
            filepath = os.path.join(in_dir, filename)
            data = pl.read_ipc(filepath)
            trajectories = build_trajectories(data)
            if trajectories is not None:
                out_filepath = os.path.join(out_dir, filename.split(".")[0])
                np.save(out_filepath, trajectories)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to the AIS data CSV file")
    parser.add_argument(
        "--output-dir", default="data/mmsi/", help="Output directory for grouped data"
    )
    args = parser.parse_args()

    # Load data and process with progress bar
    data = load_data(args.path)
    group_data(data, args.output_dir)
    build_dataset

    print(f"Data '{args.path}' grouped by MMSI and saved to '{args.output_dir}'")
