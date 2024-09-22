# import ctypes

# lib = ctypes.CDLL("sse-server/target/debug/libsse_server.so")

# lib.start_sse_server()

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# data = pd.read_feather('data/mmsi/211240870.feather')

data = np.load('data/trajectories/211534760.npy')
print(data)
print(data.shape)
# print(data)
# print(data.shape)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

plt.figure(figsize=(9, 9))


plt.scatter(data[0], data[1], color="red")
plt.show()

exit()

# Get unique MMSI values
unique_mmsi = data['mmsi'].unique()

# Create a colormap
colors = cm.rainbow(np.linspace(0, 1, len(unique_mmsi)))

# Plot Longitude and Latitude for each MMSI in different colors
for mmsi, color in zip(unique_mmsi, colors):
    mmsi_data = data[data['mmsi'] == mmsi]
    plt.scatter(mmsi_data['longitude'], mmsi_data['latitude'], color=color, label=f"mmsi {mmsi}")

plt.xlabel("lat")
plt.ylabel("lng")
plt.legend()
plt.show()