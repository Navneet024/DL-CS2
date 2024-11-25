import streamlit as st
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Function to load the model
def load_saved_model(model_path):
    return load_model(model_path)

# Function to preprocess the image for the model
def preprocess_image(img):
    # Resize the image to match the input size of EfficientNetB0
    img = img.resize((224, 224))
    # Convert image to numpy array
    img_array = np.array(img)
    # Add batch dimension (since the model expects a batch of images)
    img_array = np.expand_dims(img_array, axis=0)
    # Preprocess image for EfficientNet
    img_array = preprocess_input(img_array)
    return img_array

# Streamlit app
st.title("EfficientNet Model - Image Classification")

# Sidebar options
option = st.sidebar.selectbox("Choose an action", ["Load Model", "Classify Image"])

if option == "Load Model":
    st.header("Load a Saved Model")
    uploaded_model = st.file_uploader("Upload the model file (.h5)", type=["h5"])
    if uploaded_model:
        # Load the model
        model = load_saved_model(uploaded_model)
        st.success("Model loaded successfully!")
        st.session_state['model'] = model  # Store model in session state

elif option == "Classify Image":
    st.header("Classify an Image")
    
    if 'model' not in st.session_state:
        st.warning("Please load a model first!")
    else:
        # Load the uploaded image
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        
        if uploaded_image:
            # Open the image with PIL
            img = Image.open(uploaded_image)
            st.image(img, caption="Uploaded Image", use_column_width=True)

            # Preprocess the image
            img_array = preprocess_image(img)

            # Make prediction
            model = st.session_state['model']  # Retrieve model from session state
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)

            # Display prediction
            st.write(f"Predicted class index: {predicted_class[0]}")
            
            # Optionally, display the class name if you have a class label mapping
            # For example, if you have a list of class names:
            class_names = ['Class1', 'Class2', 'Class3', 'Class4', 'Class5', 'Class6', 'Class7', 'Class8', 'Class9', 'Class10']  # Example, update according to your class labels
            st.write(f"Predicted class name: {class_names[predicted_class[0]]}")
