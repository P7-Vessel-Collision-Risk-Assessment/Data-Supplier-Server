import argparse
from concurrent.futures import ProcessPoolExecutor
import os
import subprocess
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

# Preprocessing parameters
TIME_THRESHOLD = pd.Timedelta(minutes=30)  # Maximum allowed time gap between consecutive points
INTERPOLATION_INTERVAL = pd.Timedelta(minutes=1) 
SCALE_RANGE = (-1, 1)  
CHUNK_SIZE = 10**6
LAST_TIMESTAMPS = {}

def count_lines(path: str) -> int:
    return int(subprocess.check_output(["wc", "-l", path]).split()[0])

def process_group(name, group, dir):
    os.makedirs(dir, exist_ok=True)

    df_interpolated = interpolate_trajectory(group)

    scaler = MinMaxScaler(feature_range=SCALE_RANGE)

    df_normalized = normalize_trajectories(df_interpolated, scaler)

    file_path = f"{dir}/{name[0]}:{name[1]}.feather"

    if os.path.exists(file_path):
        combined_data = pd.read_feather(file_path)
        combined_data = pd.concat([combined_data, df_normalized])
        combined_data.to_feather(file_path)
    else:
        df_normalized.to_feather(file_path)

def extract_trajectories(df: pd.DataFrame, time_threshold=TIME_THRESHOLD):
    df = df.sort_values(by=['mmsi', 'timestamp'])

    # for mmsi in df["mmsi"].unique():
    #     if mmsi in LAST_TIMESTAMPS:
    #         first_time = df.loc[df["mmsi"] == mmsi, "timestamp"].iloc[0]
    #         if first_time - LAST_TIMESTAMPS[mmsi] > time_threshold:
    #             df.loc[df["mmsi"] == mmsi, "new_segment"] = True
    # for mmsi in df["mmsi"].unique():
    #     LAST_TIMESTAMPS[mmsi] = df.loc[df["mmsi"] == mmsi, "timestamp"].iloc[-1]

    df["time_diff"] = df.groupby("mmsi")["timestamp"].diff()
    df["segment"] = (df["time_diff"] > time_threshold).add(1)  # New segment if time gap exceeds threshold
    return df

def interpolate_trajectory(group: pd.DataFrame, interval=INTERPOLATION_INTERVAL):
    group.set_index('timestamp', inplace=True)
    idx = pd.date_range(group.index.min(), group.index.max(), freq=interval)
    group = group.reindex(idx)
    # group = group.resample("1min").mean(numeric_only=True).interpolate(method="linear")
    # Interpolate missing values
    group["longitude"] = group["longitude"].interpolate(method="linear")
    group["latitude"] = group["latitude"].interpolate(method="linear") 
    group["sog"] = group["sog"].interpolate(method="linear") 
    group["cog"] = group["cog"].interpolate(method="linear") 
    
    return group.reset_index().rename(columns={'index': 'timestamp'})

def normalize_trajectories(df, scaler):
    df[['longitude', 'latitude', 'sog', 'cog']] = scaler.fit_transform(df[['longitude', 'latitude', 'sog', 'cog']])
    return df


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to the AIS data CSV file")
    parser.add_argument(
        "--output-dir", default="data/mmsi/", help="Output directory for grouped data"
    )
    args = parser.parse_args()
    df = pd.read_csv('data/aisdk-2024-09-11.csv', parse_dates=["# Timestamp"], chunksize=CHUNK_SIZE)

    total_chunks = count_lines(args.path) // CHUNK_SIZE

    pbar = tqdm(total=total_chunks, desc="Processing chunks")

    with ProcessPoolExecutor() as executor:

        chunk: pd.DataFrame
        for chunk in df:
            pd.set_option("display.max_rows", None)
            # pd.set_option("display.max_columns", None)

            chunk.rename(columns={"# Timestamp": "timestamp"}, inplace=True)
            chunk.rename(lambda x:x.lower(), axis="columns", inplace=True)
            chunk = chunk.loc[chunk["type of mobile"] == "Class A"]
            chunk.drop_duplicates(subset=["mmsi", "timestamp"], inplace=True)
            chunk.dropna(subset=["longitude", "latitude", "sog", "cog"], inplace=True)

            trajectories = extract_trajectories(chunk)

            trajectories_grouped = trajectories.groupby(["mmsi", "segment"])

            futures = [
                executor.submit(process_group, name, group, args.output_dir)
                for name, group in trajectories_grouped 
            ]

            for future in futures:
                future.result()

            pbar.update(1)

    pbar.close()

    print(f"Data '{args.path}' processed and saved to '{args.output_dir}'")


