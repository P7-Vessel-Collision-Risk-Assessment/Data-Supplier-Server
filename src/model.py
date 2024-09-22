import torch
import torch.nn as nn

class VRAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, output_size, encode_layers, decode_layers):
        super(VRAE, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, latent_size, encode_layers)
        self.decoder = Decoder(latent_size, hidden_size, output_size, decode_layers)

    def forward(self, x):
        # Encode the input sequence
        mu, logvar = self.encoder(x)

        # Reparametrize to get latent vector z
        z = reparametrize(mu, logvar)

        # Decode to reconstruct the input sequence
        recon_x = self.decoder(z, x.size(1))

        return recon_x, mu, logvar

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, num_layers=1):
        super(Encoder, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc_mu = nn.Linear(hidden_size * 2, latent_size)  # For the mean vector
        self.fc_logvar = nn.Linear(hidden_size * 2, latent_size)  # For the log variance vector

    def forward(self, x):
        # x: [batch_size, sequence_length, input_size]
        _, h = self.gru(x)  # h: [2, batch_size, hidden_size] -> Bidirectional GRU output
        h = torch.cat((h[-2], h[-1]), dim=1)  # Concatenate forward and backward hidden states

        # Mean and log variance vectors
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_size, hidden_size, output_size, num_layers=1):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.fc_hidden = nn.Linear(latent_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc_output = nn.Linear(hidden_size, output_size)

    def forward(self, z, seq_len):
        # z: [batch_size, latent_size]
        hidden = self.fc_hidden(z).unsqueeze(0).repeat(self.num_layers, 1, 1)  # Hidden state for GRU: [2, batch_size, hidden_size]

        # Initialize inputs (for teacher forcing, start with zeros)
        inputs = torch.zeros((z.size(0), seq_len, hidden.size(-1)), device=z.device)
        
        # GRU sequence generation
        outputs, _ = self.gru(inputs, hidden)
        outputs = self.fc_output(outputs)
        return outputs  # Reconstructed sequence: [batch_size, sequence_length, output_size]

def reparametrize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std
