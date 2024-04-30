# VAEs and GANs with S4 architecture

Three different models/methods are run for the image generation task on MNIST and CIFAR10 dataset.
- The folder $$baseline/$$ consists of all the files (python/jupyter files and trained model weights) of image generation with DCGAN architecture.
- The folder $$S4-GAN$$ consists of all the files (python/jupyter files and trained model weights) of image generation with S4-GAN architecture.
- The folder $$S4-VAE$$ consists of all the files (python/jupyter files and trained model weights) of image generation with S4-VAE architecture.

[Frechet Inception Distance](https://arxiv.org/abs/1706.08500) has been used as the evaluation metric. The lower the FID, better is the image generation.
We see a sharp decrease in FID score using the S4-VAE architecture when compared to the DCGAN baseline. This shows that S4 models are highly superior compared to traditional architectures.

The FID scores are shown below.
- DCGAN - MNIST: 63.66
- S4-VAE - MNIST: **22.34**
- DCGAN - CIFAR10: 192.28
- S4-VAE - CIFAR10: **8.17**
