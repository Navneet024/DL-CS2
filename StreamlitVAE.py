import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

# Define your VAE model (same as the one used for training)
class VAE(nn.Module):
    def __init__(self, beta=1.0):
        super(VAE, self).__init__()
        self.beta = beta  # Beta for controlling KL divergence weight

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Latent space layer (mu, log_var)
        self.fc_mu = nn.Linear(256 * 4 * 4, 512)
        self.fc_log_var = nn.Linear(256 * 4 * 4, 512)

        # Decoder
        self.decoder_fc = nn.Linear(512, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output range [-1, 1]
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = self.decoder_fc(z)
        z = z.view(-1, 256, 4, 4)
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z)
        return recon_x, mu, log_var

# Define the loss function
def vae_loss_tanh(recon_x, x, mu, log_var, beta):
    MSE = nn.functional.mse_loss(recon_x.view(-1, 3 * 64 * 64), x.view(-1, 3 * 64 * 64), reduction='sum')
    D_KL = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    total_loss = MSE + beta * D_KL
    return MSE, D_KL, total_loss

# Load models
model = VAE(beta=1.0)
model.load_state_dict(torch.load('vae_epoch_50.pth'))  # Adjust path if needed
model.eval()

# Streamlit app layout
st.title("VAE Image Generation App")

# Upload image functionality
st.sidebar.header("Upload an Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Image transformation and prediction
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Button to generate image
    if st.button("Generate Image"):
        # Generate image (using the VAE model)
        with torch.no_grad():
            recon_x, _, _ = model(image)

        # Rescale to [0, 1] range and convert to numpy
        recon_x = (recon_x.squeeze().numpy() + 1) / 2  # From [-1, 1] to [0, 1]
        recon_x = np.transpose(recon_x, (1, 2, 0))  # Convert to HxWxC format for visualization

        # Display generated image
        st.image(recon_x, caption="Generated Image", use_column_width=True)
