import argparse
import subprocess
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from pandas.io.parsers import TextFileReader
import pandas as pd
import os
import numpy as np

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

def build_trajectories(data: TextFileReader):
    data = data.drop_duplicates()

    data['timestamp'] = pd.to_datetime(data['timestamp'], format='%d/%m/%Y %H:%M:%S')

    cols = ['latitude', 'longitude', 'sog', 'cog_x', 'cog_y']

    data = data.infer_objects()

    # Normalize columns 
    data.dropna(subset=cols, inplace=True)

    # Return if dataframe is empty after removing rows with NaN values
    if data.empty:
        return None

    data.set_index('timestamp', inplace=True)

    # Resample at 1-minute intervals and interpolate. 
    # A lot of cases where we get way worse density, maybe there are better ways to do the interpolation?
    data = data.resample('1min').mean(numeric_only=True).interpolate(method = 'linear')

    data.reset_index(inplace=True)

    scaler = MinMaxScaler(feature_range=(-1, 1))

    data[cols] = scaler.fit_transform(data[cols])

    # data = trajectory_to_segments(data)

    data = np.array(data[['latitude', 'longitude', 'sog', 'cog_x', 'cog_y']].values)

    return data

def trajectory_to_segments(df, segment_length=30):
    # Ensure we have a 'Timestamp' as a datetime column for splitting
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    segments = []

    # Iterate over each segment and store it if it's 30 minutes long
    for start in range(0, len(df) - segment_length + 1, segment_length):
        segment = df.iloc[start:start + segment_length]
        if len(segment) == segment_length:
            segments.append(segment[['latitude', 'longitude', 'sog', 'cog_x', 'cog_y']].values)

    if len(segments) == 0:
        return None

    return np.array(segments)

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
                "Length"
            ]
        ].rename(columns={"# Timestamp": "Timestamp"})

        chunk = chunk[chunk['Type of mobile'] == 'Class A']
        
        chunk.columns = chunk.columns.str.lower()   

        chunk["cog_x"], chunk["cog_y"] = zip(*chunk["cog"].map(cog_to_xy))
         
         # Group by MMSI (vessel identifier)
        grouped = chunk.groupby('mmsi')

        # Process each vessel's data
        for name, group in grouped:
            if os.path.exists(f"{dir}/{name}.feather"):
                combined_data = pd.read_feather(f"{dir}/{name}.feather")
                combined_data = pd.concat([combined_data, group])                
                combined_data.to_feather(f"{dir}/{name}.feather")
                del combined_data
            else:
                group.to_feather(f"{dir}/{name}.feather")

        pbar.update(1)

    pbar.close()


def build_dataset(in_dir="data/mmsi", out_dir="data/trajectories"):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    for filename in os.listdir(in_dir):
        if filename.endswith('.feather'):
            filepath = os.path.join(in_dir, filename)
            data = pd.read_feather(filepath)
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
    build_dataset()

    print(f"Data '{args.path}' grouped by MMSI and saved to '{args.output_dir}'")

