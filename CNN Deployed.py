import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('CNN.keras')

# Define class labels
class_labels = ['drink', 'food', 'inside', 'menu', 'outside']

def predict_class(img_path):
    # Load and preprocess the image
    img = load_img(img_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = class_labels[predicted_class_index]
    predicted_probability = predictions[0][predicted_class_index]

    return predicted_class, predicted_probability

# Streamlit app
st.title("Image Classifier")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Predict the class
    img_path = uploaded_file.name
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    predicted_class, predicted_probability = predict_class(img_path)

    # Display the prediction
    st.write(f'Predicted class: {predicted_class}, Probability: {predicted_probability:.2f}')

