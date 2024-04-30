import torch
from torch import nn
from torchvision.models import inception_v3
import numpy as np
from scipy import linalg
from dc_gan import Discriminator, Generator, init_weights
import math
import scipy

class PartialInceptionNetwork(nn.Module):

    def __init__(self, transform_input=True):
        super().__init__()
        self.inception_network = inception_v3(pretrained=True)
        self.inception_network.Mixed_7c.register_forward_hook(self.output_hook)
        self.transform_input = transform_input

    def output_hook(self, module, input, output):
        # N x 2048 x 8 x 8
        self.mixed_7c_output = output

    def forward(self, x):
        """
        Args:
            x: shape (N, 3, 299, 299) dtype: torch.float32 in range 0-1
        Returns:
            inception activations: torch.tensor, shape: (N, 2048), dtype: torch.float32
        """
        assert x.shape[1:] == (3, 299, 299), "Expected input shape to be: (N,3,299,299)" +\
                                             ", but got {}".format(x.shape)
        x = x * 2 -1 # Normalize to [-1, 1]
        self.inception_network(x)

        # Output: N x 2048 x 1 x 1 
        activations = self.mixed_7c_output
        activations = torch.nn.functional.adaptive_avg_pool2d(activations, (1,1))
        activations = activations.view(x.shape[0], 2048)
        return activations


def calculate_fid(real_activations, gen_activations):

    """
    Given the activations this function returns the FID score.
    """
    real_activations_cpu = real_activations.cpu().detach().numpy()
    gen_activations_cpu = gen_activations.cpu().detach().numpy()
    mu_real = np.mean(real_activations_cpu, axis=0)
    mu_gen = np.mean(gen_activations_cpu, axis=0)
    cov_real = np.cov(real_activations_cpu, rowvar=False)
    cov_gen = np.cov(gen_activations_cpu, rowvar=False)

    diff = mu_real - mu_gen
    diff_squared = diff.dot(diff)
    prod = cov_real.dot(cov_gen)
    sqrt_prod, _ = scipy.linalg.sqrtm(prod, disp = False)

    if np.iscomplexobj(sqrt_prod):
        sqrt_prod = sqrt_prod.real
    prod = np.trace(sqrt_prod)
    fid_score = diff_squared + np.trace(cov_real) + np.trace(cov_gen) - 2*prod
    
    return fid_score


def fid_score(real_images, gen_images, batch_size, inception_network):

    """
    Given the real images this functions generates the activations and passes it on to the calculate_fid 
    function. Finally returns the fid score.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    real_activations = []
    gen_activations = []
    num_steps = int(math.ceil(float(len(real_images)) / float(batch_size)))
    net = PartialInceptionNetwork().to(device)
    net.eval()

    for i in range(num_steps):
        start_idx = i * batch_size
        end_idx = (i+1) * batch_size
        batch = real_images[start_idx:end_idx]
        batch = batch.to(device)
        with torch.no_grad():
            activations = net(batch)
            real_activations.append(activations)
    for i in range(num_steps):
        start_idx = i * batch_size
        end_idx = (i+1) * batch_size
        batch = gen_images[start_idx:end_idx]
        with torch.no_grad():
            activations = net(batch)
            gen_activations.append(activations)
    real_activations = torch.cat(real_activations, dim=0)
    gen_activations = torch.cat(gen_activations, dim=0)
    fid_score = calculate_fid(real_activations, gen_activations)
    return fid_score

