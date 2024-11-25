import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
from PIL import Image

# Function to load the model
def load_saved_model(model_path):
    return load_model(model_path)

# Function to preprocess the image for the model
def preprocess_image(img):
    # Resize the image to match the input size of EfficientNetB0
    img = img.resize((224, 224))
    
    # Convert image to numpy array and ensure RGB format
    img_array = np.array(img.convert('RGB'))
    
    # Ensure the image has 3 channels (RGB)
    if img_array.shape[-1] != 3:
        raise ValueError("Input image must have 3 channels (RGB).")
    
    # Add batch dimension (since the model expects a batch of images)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Preprocess image for EfficientNet
    img_array = preprocess_input(img_array)
    
    return img_array

# Streamlit app
st.title("EfficientNet Model - Image Classification")

# Sidebar options
option = st.sidebar.selectbox("Choose an action", ["Classify Image"])

# Load the model at the start
model = load_saved_model('Efficient_Net.h5')

if option == "Classify Image":
    st.header("Classify an Image")
    
    # Load the uploaded image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        
    if uploaded_image:
        # Open the image with PIL
        img = Image.open(uploaded_image)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        try:
            # Preprocess the image
            img_array = preprocess_image(img)

            # Make prediction
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)

            # Display prediction
            st.write(f"Predicted class index: {predicted_class[0]}")
            
            # Optionally, display the class name if you have a class label mapping
            # Replace this with your actual class names (as defined during training)
            class_names = ['inside', 'outside', 'drink', 'food', 'other']  # Example, replace with actual class names
            st.write(f"Predicted class name: {class_names[predicted_class[0]]}")
        
        except ValueError as e:
            st.error(f"Error processing image: {e}")
