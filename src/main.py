from data import reader

if __name__ == "__main__":
    data = reader("data/mmsi")
    for df in data:
        print(df)
