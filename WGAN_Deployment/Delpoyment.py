import streamlit as st
import torch
from torchvision.utils import make_grid, save_image
from PIL import Image
import os
import io

# Path to the trained model
model_path = r"C:\Users\Asus\OneDrive\Desktop\Final_Project\wgan_models-50.pth"  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the Generator class (same as the training script)
class Generator(torch.nn.Module):
    def __init__(self, latent_dim=128):
        super(Generator, self).__init__()
        self.init_size = 64 // 4
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 128 * self.init_size ** 2),
            torch.nn.BatchNorm1d(128 * self.init_size ** 2),
            torch.nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(128, 128, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(128, 64, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(64, 3, 3, stride=1, padding=1),
            torch.nn.Tanh(),
        )

    def forward(self, z):
        out = self.fc(z)
        out = out.view(out.size(0), 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# Load the trained Generator model
@st.cache_resource
def load_generator():
    generator = Generator().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    generator.load_state_dict(checkpoint['generator'])
    generator.eval()
    return generator

generator = load_generator()

# Streamlit App UI
st.title("WGAN Image Generator")
st.sidebar.header("Settings")

# User inputs
latent_dim = 128
num_images = st.sidebar.slider("Number of images to generate", min_value=1, max_value=10, value=1)
seed = st.sidebar.number_input("Random seed (optional)", value=42, step=1)

# Generate button
if st.sidebar.button("Generate"):
    # Set random seed for reproducibility
    torch.manual_seed(seed)

    # Generate images
    st.subheader("Generated Images")
    z = torch.randn(num_images, latent_dim, device=device)
    with torch.no_grad():
        generated_images = generator(z).cpu()
        generated_images = (generated_images + 1) / 2  

    # Display images
    for i in range(num_images):
        img_tensor = generated_images[i]
        img = Image.fromarray((img_tensor.permute(1, 2, 0).numpy() * 255).astype("uint8"))
        st.image(img, caption=f"Generated Image {i+1}", use_column_width=True)

        # Download button
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        st.download_button(
            label=f"Download Image {i+1}",
            data=buf,
            file_name=f"generated_image_{i+1}.png",
            mime="image/png",
        )
