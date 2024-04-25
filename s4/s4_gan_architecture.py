import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from s4d import S4D

if tuple(map(int, torch.__version__.split('.')[:2])) == (1, 11):
    print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
    dropout_fn = nn.Dropout
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d

class S4Gen(nn.Module):

    def __init__(
        self,
        d_input,
        d_output=10,
        d_model=256,
        n_layers=4,
        dropout=0.2,
        prenorm=False,
    ):
        super().__init__()

        self.prenorm = prenorm

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4D(d_model, dropout=dropout, transposed=True, lr=min(0.001, 0.01))
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(dropout_fn(dropout))

        # Linear decoder
        self.decoder = nn.Linear(1, d_output)

        self.channel_reducer = nn.Linear(d_model, 1)
        
        self.d_output = d_output


    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        print("X shape",x.shape)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            print(z.shape)
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        #x = x.transpose(-1, -2)

        # Pooling: average pooling over the sequence length
        #x = x.mean(dim=1)
        #print(x.shape)

        # Decode the outputs
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)
        x = x.view(x.shape[0], x.shape[2], x.shape[1])
        x = self.channel_reducer(x)
        x = x.transpose(-1, -2)
        #print(x.shape)
        x = x.view(x.shape[0], x.shape[1], int(math.sqrt(self.d_output)), int(math.sqrt(self.d_output)))

        return x
    
class S4Disc(nn.Module):
    def __init__(self,
                 channels, 
                 d_output, 
                 n_layers,
                 d_model = 256,
                 dropout = 0.2,  
                 prenorm = False):
        super().__init__()
        self.prenorm = prenorm

        # Linear Encoder 
        self.encoder = nn.Linear(channels, d_model) # (B, L, d_input) -> (B, L, d_model)

        # Stack S4 Layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for _ in range(n_layers):
            self.s4_layers.append(
                S4D(d_model, dropout=dropout, transposed=True, lr=min(0.001, 0.01))
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(dropout_fn(dropout))
        
        # Linear decoder 
        self.decoder_layer1 = nn.Linear(4096, d_output)
        self.decoder_layer2 = nn.Linear(256, d_output)
        self.sigmoid = nn.Sigmoid()

        self.d_output = d_output

    def forward(self, x):
        """
        Input shape is (N, C, B, L)
        """
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)
        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L) 
        
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)
            

        #x = x.transpose(-1, -2)

        # Pooling: average pooling over the sequence length
        #x = x.mean(dim=1)
        #print(x.shape)

        # Decode the outputs
        x = self.decoder_layer1(x)  
        print(x.shape) 
        x = x.squeeze()
        x = self.decoder_layer2(x)
        x = x.squeeze()

        x = self.sigmoid(x)
        

        return x


class Embedding(nn.Module):
    def __init__(self, num_embeddings, features):
        super(Embedding, self).__init__()
        self.embed = nn.Embedding(num_embeddings, features, padding_idx=0)

    def forward(self, x):
        x = x[..., 0]  # Assuming x has a last dimension that we need to index
        y = self.embed(x)
        return y



if __name__ == '__main__':
    # Generator Model
    print('==> Building generator..')
    model = S4Gen(
        d_input=100,
        d_output=64*64,
        d_model=256,
        n_layers=4,
        dropout=0.2,
        prenorm=False,
    )
    

    N, in_channels, H, W = 8, 3, 64, 64
    noise_dim = 100
    x = torch.randn((128, 1, 100))
    y = model(x)
    #print(y.shape)

    # Discriminator Model test
    print('==>Building discriminator..')
    model = S4Disc(channels=1,
                   d_output=1,
                   n_layers=4,
                   d_model=256,
                   dropout=0.2,
                   prenorm=False)
    
    N, in_channels, height, width = 128, 1, 64, 64
    x = torch.randn((N, height*width, in_channels))
    y = model(x)
    print(y.shape)
