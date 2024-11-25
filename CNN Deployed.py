import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from io import BytesIO

# Load the trained model
model = tf.keras.models.load_model('CNN.keras')

# Define class labels
class_labels = ['drink', 'food', 'inside', 'menu', 'outside']

def predict_class(img):
    # Preprocess the image: resize and normalize
    img = img_to_array(img) / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(img)
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
    img = load_img(uploaded_file, target_size=(128, 128))  # Resize image to match the input shape of the model
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Predict the class using the in-memory image
    predicted_class, predicted_probability = predict_class(img)

    # Display the prediction
    st.write(f'Predicted class: {predicted_class}')
    st.write(f'Probability: {predicted_probability:.2f}')
