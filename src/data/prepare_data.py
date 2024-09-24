import argparse
import datetime
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

def load_data(path: str) -> pl.LazyFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pl.scan_csv(path, with_column_names=lambda cols: [col.lower() for col in cols])

def build_trajectories(data: pl.DataFrame):
    data = data.unique(subset=None, keep="first")

    # Convert timestamp column to datetime
    data = data.with_columns([
        pl.col("timestamp").str.strptime(pl.Datetime, format="%d/%m/%Y %H:%M:%S").alias("timestamp")
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
    ).agg([pl.col(cols).mean()]).with_columns([pl.col(col).interpolate() for col in cols])



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

def group_data_in_memory(data: pl.LazyFrame, dir: str):
    os.makedirs(dir, exist_ok=True)

    data = (
        data.rename({"# timestamp": "timestamp"})
        .filter(pl.col("type of mobile") == "Class A")
        .with_columns([
            pl.col("cog").radians().sin().alias("cog_x"),
            pl.col("cog").radians().cos().alias("cog_y")
        ])
        .drop("cog")
        .collect(streaming=True)
        .group_by("mmsi")
    )

    vessel_count = data.n_unique().count().to_numpy()[0][0]

    pbar = tqdm(total=vessel_count, desc="Processing chunks")
    
    vessel_data: pl.DataFrame  
    for name, vessel_data in data:
        file_path = f"{dir}/{name[0]}.feather"    

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
    group_data_in_memory(data, args.output_dir)
    build_dataset()

    print(f"Data '{args.path}' grouped by MMSI and saved to '{args.output_dir}'")
