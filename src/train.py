import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence

from model import VRAE

class TrajectoryDataset(Dataset):
    def __init__(self, data_dir):
        self.data = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('npy')]
        self.trajectories = []

        # Load each vessel's data and flatten the trajectory dimension
        for file in self.data:
            vessel_data = np.load(file)  # Load data in shape [trajectory_idx, trajectory_length, trajectory_features]
            for traj in vessel_data:
                self.trajectories.append(traj)

    def __len__(self):
        return len(self.trajectories)
        
    def __getitem__(self, idx):
        trajectory = self.trajectories[idx]
        return torch.tensor(trajectory, dtype=torch.float32) 

def loss_function(recon_x, x, mu, logvar):
    # Reconstruction loss (MSE)
    recon_loss = nn.MSELoss()(recon_x, x)

    # KL Divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl_loss

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model hyperparameters
input_size = 4  # Number of features in input (lat, lng, sog, cog_x, cog_y)
hidden_size = 50
latent_size = 50
output_size = 4
encode_layers = 3
decode_layers = 2

batch_size = 64
num_epochs = 10
learning_rate = 3e-4

data_dir = 'data/trajectories'
dataset = TrajectoryDataset(data_dir)

dataloader = DataLoader(dataset, batch_size=batch_size)

# Initialize the model
model = VRAE(input_size, hidden_size, latent_size, output_size, encode_layers, decode_layers).to(device)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for batch in dataloader:
        batch = batch.to(device)
        recon_data, mu, logvar = model(batch)
        
        loss = loss_function(recon_data, batch, mu, logvar)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():

    batch = np.load("data/trajectories/211534760.npy") 
    batch = torch.tensor(batch, dtype=torch.float32)
    batch = batch.to(device)
    # batch = next(iter(dataloader))
    # batch = batch.to(device)

    recon_data, mu, logvar = model(batch)

    batch = batch.cpu().numpy().reshape(-1, batch.shape[-1])
    recon_data = recon_data.cpu().numpy().reshape(-1, batch.shape[-1])

    plt.figure(figsize=(8,8))

    plt.scatter(batch[:,0], batch[:,1], color='blue', label="Original")
    plt.scatter(recon_data[:,0], recon_data[:,1], color='green', label="Reconstructed")

    plt.xlabel("lat")
    plt.ylabel("lng")
    plt.legend()
    plt.show()