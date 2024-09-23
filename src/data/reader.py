import os
import pandas as pd


def reader(dir: str):
    """Creates a generator to read data from the dir of feather files

    Args:
        dir (str): The directory to read the feather files from

    Yields:
        dataframe (pandas.DataFrame): A DataFrame read from a feather file
    """
    for file in os.listdir(dir):
        if file.endswith(".feather"):
            yield pd.read_feather(f"{dir}/{file}")
