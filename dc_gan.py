import torch
import torch.nn as nn

def init_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

class Generator(nn.Module):
    def __init__(self, noise_dim, channels_img, features_g):
        super(Generator, self).__init__()
        self.generator = nn.Sequential(
            self.layer(noise_dim, features_g*16, 4, 1, 0), 
            self.layer(features_g*16, features_g*8, 4, 2, 1),
            self.layer(features_g*8, features_g*4, 4, 2, 1), 
            self.layer(features_g*4, features_g*2, 4, 2, 1), 
            nn.ConvTranspose2d(features_g*2, channels_img, 4, 2, 1),
            nn.Tanh() 
        )
    
    def layer(self, in_channels, out_channels, kernel_size, stride, padding):
        layer  = nn.Sequential(
              nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU()
        )
        return layer

    def forward(self, x):
        return self.generator(x)



class Discriminator(nn.Module):

    def __init__(self, channels_img, features_d):
        #input image dimensions: (N, channels_img, 64, 64)
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride =2 , padding =1, ), 
            self.layer(features_d, 2*features_d, kernel_size=4, stride=2, padding=1,),
            self.layer(2*features_d, 4*features_d, kernel_size=4, stride=2, padding=1,), 
            self.layer(4*features_d, 8*features_d, kernel_size=4, stride=2, padding=1,),
            nn.Conv2d(8*features_d, 1, kernel_size=4, stride=2, padding = 0), 
            nn.Sigmoid(), 
        )

    def layer(self, in_channels, out_channels, kernel_size, stride, padding):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding), 
            nn.BatchNorm2d(out_channels), 
            nn.LeakyReLU(0.2) #slope given in the dcgan paper is 0.2
        )
        return layer
    
    def forward(self, x):
        return self.discriminator(x)
    

def test_case():
    N, in_channels, H, W = 8, 3, 64, 64
    noise_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, 8)
    assert disc(x).shape == (N, 1, 1, 1), "Discriminator test failed"
    gen = Generator(noise_dim, in_channels, 8)
    z = torch.randn((N, noise_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"
    print("Success, tests passed!")