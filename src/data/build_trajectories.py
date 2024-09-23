import os
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import argparse


def build_trajectories(data: pd.DataFrame) -> np.ndarray:
    data = data.drop_duplicates()

    data.loc[:, "timestamp"] = pd.to_datetime(
        data["timestamp"], format="%d/%m/%Y %H:%M:%S"
    )

    cols = ["latitude", "longitude", "sog", "cog_x", "cog_y"]

    data = data.infer_objects()

    # Normalize columns
    data.dropna(subset=cols, inplace=True)

    # Return if dataframe is empty after removing rows with NaN values
    if data.empty:
        return np.array([])

    data.set_index("timestamp", inplace=True)

    # Resample at 1-minute intervals and interpolate.
    # A lot of cases where we get way worse density, maybe there are better ways to do the interpolation?
    data = data.resample("1min").mean(numeric_only=True).interpolate(method="linear")

    data.reset_index(inplace=True)

    scaler = MinMaxScaler(feature_range=(-1, 1))

    data[cols] = scaler.fit_transform(data[cols])

    # data = trajectory_to_segments(data)

    return np.array(data[["latitude", "longitude", "sog", "cog_x", "cog_y"]].values)


def trajectory_to_segments(df, segment_length=30):
    # Ensure we have a 'Timestamp' as a datetime column for splitting
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    segments = []

    # Iterate over each segment and store it if it's 30 minutes long
    for start in range(0, len(df) - segment_length + 1, segment_length):
        segment = df.iloc[start : start + segment_length]
        if len(segment) == segment_length:
            segments.append(
                segment[["latitude", "longitude", "sog", "cog_x", "cog_y"]].values
            )

    if len(segments) == 0:
        return None

    return np.array(segments)


def build_dataset(in_dir="data/mmsi", out_dir="data/trajectories"):
    os.makedirs(out_dir, exist_ok=True)

    total_files = len(os.listdir(in_dir))

    pbar = tqdm(total=total_files, desc="Building trajectories")

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_trajectory, in_dir, out_dir, filename)
            for filename in os.listdir(in_dir)
        ]

        for future in futures:
            pbar.update(1)
            future.result()

    pbar.close()


def process_trajectory(in_dir, out_dir, filename):
    if filename.endswith(".feather"):
        filepath = os.path.join(in_dir, filename)
        data = pd.read_feather(filepath)
        trajectories = build_trajectories(data)
        if len(trajectories) > 0:
            out_filepath = os.path.join(out_dir, filename.split(".")[0])
            np.save(out_filepath, trajectories)
            # pbar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", "-i", type=str, default="data/mmsi")
    parser.add_argument("--output-dir", "-o", type=str, default="data/trajectories")
    args = parser.parse_args()

    build_dataset(args.input_dir, args.output_dir)
