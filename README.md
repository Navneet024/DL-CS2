# Gans on yelp dataset
Deep Learning 

# Generative Models for Image Generation using Yelp Dataset

This repository contains implementations of several generative models to generate images using the Yelp dataset. These models are:

## 1. DCGAN (Deep Convolutional GAN)
A type of GAN that uses deep convolutional layers to generate realistic images from random noise. The generator tries to generate convincing images, and the discriminator tries to distinguish between real and generated images.

- **Generator**: Takes random noise and generates fake images.
- **Discriminator**: Classifies images as real or fake.
- **Use Case**: Image generation from random noise.

## 2. WGAN (Wasserstein GAN)
A modification of the GAN that uses the Wasserstein distance to provide better gradient flow during training and stabilize the learning process. It uses a gradient penalty to enforce Lipschitz continuity.

- **Wasserstein Loss**: More stable training by using Wasserstein distance.
- **Gradient Penalty**: Regularizes the model to avoid instability.
- **Use Case**: More stable training for high-dimensional datasets like images.

## 3. VAE (Variational Autoencoder)
A generative model that learns a probabilistic mapping from a high-dimensional data space to a lower-dimensional latent space. It generates new data by sampling from the learned latent distribution.

- **Encoder**: Maps input data to a probabilistic distribution in the latent space.
- **Decoder**: Generates new data from the latent space.
- **Use Case**: Data generation and learning useful data representations.

## 4. cGAN (Conditional GAN)
An extension of GANs where both the generator and discriminator are conditioned on class labels. This allows generation of images that correspond to specific categories, such as `food`, `drink`, `inside`, or `outside`.

- **Generator**: Takes both noise and a class label as input to generate class-specific images.
- **Discriminator**: Classifies images based on both the image and its class label.
- **Use Case**: Generate images based on specific conditions or labels.

## Installation

Clone the repo and install dependencies:

```bash
git clone <repo_url>
cd <repo_directory>
pip install -r requirements.txt


## Evaluation:
- Inception Score
- Frechet Inception Distance (FID)

