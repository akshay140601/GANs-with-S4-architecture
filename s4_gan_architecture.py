import torch
import torch.nn as nn
import torch.nn.functional as F

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
        learning_rate,
        d_output=1,
        d_model=256,
        n_layers=4,
        dropout=0.2,
        prenorm=False
    ):
        super().__init__()

        self.prenorm = prenorm

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = nn.ConvTranspose2d(d_input, d_model, kernel_size=64, stride=4, padding=0, bias=False)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4D(d_model, dropout=dropout, transposed=True, lr=min(0.001, learning_rate))
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(dropout_fn(dropout))

        # Linear decoder
        self.decoder = nn.Conv2d(d_model, d_output, kernel_size = 1)

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        #print(x.shape)
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        #x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            #print(z.shape)
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            #z = dropout(z)

            # Residual connection
            #print(x.shape)
            #print(z.shape)
            x = z + x

            #if not self.prenorm:
                # Postnorm
                #x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        #x = x.transpose(-1, -2)

        # Pooling: average pooling over the sequence length
        #x = x.mean(dim=1)

        #Decoder
        x = self.decoder(x)

        #x = F.adaptive_avg_pool2d(x, (1, 1))

        # Decode the outputs
        #x = self.decoder(x)  # (B, d_model) -> (B, d_output)

        return x
    
class S4Disc(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        pass

    def forward(self, x):
        pass

# Model
'''print('==> Building model..')
model = S4Gen(
    d_input=100,
    d_output=1,
    d_model=256,
    n_layers=4,
    dropout=0.2,
    prenorm=False,
)

N, in_channels, H, W = 8, 3, 64, 64
noise_dim = 100
x = torch.randn((128, 100, 1, 1))
y = model(x)
print(y.shape)'''