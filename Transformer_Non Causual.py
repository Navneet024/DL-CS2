import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Define the Positional Encoding Layer (Custom Layer)
class PositionalEncodingLayer(layers.Layer):
    def __init__(self, max_len, d_model):
        super(PositionalEncodingLayer, self).__init__()
        self.max_len = max_len
        self.d_model = d_model

    def get_config(self):
        config = super(PositionalEncodingLayer, self).get_config()
        config.update({"max_len": self.max_len, "d_model": self.d_model})
        return config

    def call(self, inputs):
        position = np.arange(self.max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))
        pos_embedding = np.zeros((self.max_len, self.d_model))
        pos_embedding[:, 0::2] = np.sin(position * div_term)
        pos_embedding[:, 1::2] = np.cos(position * div_term)
        pos_embedding = np.expand_dims(pos_embedding, axis=0)
        return inputs + pos_embedding

# Non-Causal Transformer Model Definition with L2 Regularization
def build_non_causal_transformer(vocab_size, max_len, num_classes):
    inputs = layers.Input(shape=(max_len,))

    # Embedding layer
    embedding_layer = layers.Embedding(vocab_size, 128)(inputs)

    # Positional Encoding Layer (Custom Layer)
    x = PositionalEncodingLayer(max_len, 128)(embedding_layer)

    # First Transformer Encoder Layer
    transformer = layers.MultiHeadAttention(
        num_heads=4, key_dim=128, kernel_regularizer=l2(0.01)
    )(x, x)  # Non-causal, attends to all tokens
    transformer = layers.LayerNormalization(epsilon=1e-6)(transformer + x)  # Add & Normalize

    # Feed-Forward Network (MLP)
    ff = layers.Dense(512, activation='relu', kernel_regularizer=l2(0.01))(transformer)
    ff = layers.Dropout(0.6)(ff)  # Dropout layer
    ff = layers.Dense(128, kernel_regularizer=l2(0.01))(ff)
    x = layers.LayerNormalization(epsilon=1e-6)(ff + transformer)

    # Additional Transformer Layer to capture more complex patterns
    transformer_2 = layers.MultiHeadAttention(
        num_heads=4, key_dim=128, kernel_regularizer=l2(0.01)
    )(x, x)  # Second Transformer Layer
    transformer_2 = layers.LayerNormalization(epsilon=1e-6)(transformer_2 + x)  # Add & Normalize

    # Feed-Forward Network after second Transformer Layer
    ff_2 = layers.Dense(512, activation='relu', kernel_regularizer=l2(0.01))(transformer_2)
    ff_2 = layers.Dropout(0.6)(ff_2)
    ff_2 = layers.Dense(128, kernel_regularizer=l2(0.01))(ff_2)
    x = layers.LayerNormalization(epsilon=1e-6)(ff_2 + transformer_2)

    # Optional: Add third Transformer layer for even deeper learning (optional based on your problem size)
    transformer_3 = layers.MultiHeadAttention(
        num_heads=4, key_dim=128, kernel_regularizer=l2(0.01)
    )(x, x)  # Third Transformer Layer
    transformer_3 = layers.LayerNormalization(epsilon=1e-6)(transformer_3 + x)

    # Feed-Forward after third Transformer Layer
    ff_3 = layers.Dense(512, activation='relu', kernel_regularizer=l2(0.01))(transformer_3)
    ff_3 = layers.Dropout(0.6)(ff_3)
    ff_3 = layers.Dense(128, kernel_regularizer=l2(0.01))(ff_3)
    x = layers.LayerNormalization(epsilon=1e-6)(ff_3 + transformer_3)

    # Global Average Pooling (to reduce sequence dimension)
    x = layers.GlobalAveragePooling1D()(x)

    # Dropout before the final output layer
    x = layers.Dropout(0.6)(x)  # Increased Dropout Rate

    # Output layer for sentiment classification (softmax for 3 classes)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    # Model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Streamlit App Interface
st.title("Non-Causal Transformer Model Training")

# Sidebar: Upload data and select options
st.sidebar.header("Upload your Data")
uploaded_train = st.sidebar.file_uploader("Upload Training Data (CSV)", type=["csv"])
uploaded_test = st.sidebar.file_uploader("Upload Test Data (CSV)", type=["csv"])

if uploaded_train and uploaded_test:
    # Load datasets
    train_df = pd.read_csv(uploaded_train)
    test_df = pd.read_csv(uploaded_test)

    st.sidebar.write(f"Training Data Shape: {train_df.shape}")
    st.sidebar.write(f"Test Data Shape: {test_df.shape}")

    # Assuming the last column is the target label and other columns are features
    X_train = train_df.iloc[:, :-1].values
    y_train = train_df.iloc[:, -1].values
    X_test = test_df.iloc[:, :-1].values
    y_test = test_df.iloc[:, -1].values

    # Example of text preprocessing if needed (modify as per your data)
    # Here, we assume the data needs padding
    max_sequence_length = 100  # Adjust as needed
    X_train_padded = pad_sequences(X_train, maxlen=max_sequence_length)
    X_test_padded = pad_sequences(X_test, maxlen=max_sequence_length)

    # One-hot encode the labels (modify if necessary)
    y_train_encoded = to_categorical(y_train)
    y_test_encoded = to_categorical(y_test)

    # Build model (adjust input/output dimensions based on your data)
    vocab_size = 10000  # Adjust vocabulary size based on your dataset
    model = build_non_causal_transformer(vocab_size=vocab_size, max_len=max_sequence_length, num_classes=y_train_encoded.shape[1])

    st.sidebar.header("Train the Model")

    # Define EarlyStopping and ReduceLROnPlateau
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)

    # Training button
    if st.sidebar.button("Train Model"):
        with st.spinner("Training the model..."):
            history = model.fit(
                X_train_padded, 
                y_train_encoded, 
                epochs=5, 
                batch_size=64, 
                validation_data=(X_test_padded, y_test_encoded), 
                callbacks=[early_stopping, reduce_lr]
            )
        
        # Save the model after training
        model_save_path = 'Transformer.h5'
        model.save(model_save_path)
        st.success(f"Model trained and saved at {model_save_path}")
        
        # Allow user to download the saved model
        with open(model_save_path, "rb") as f:
            st.download_button(
                label="Download Trained Model",
                data=f,
                file_name=model_save_path,
                mime="application/octet-stream"
            )

        # Optionally, display training history
        st.write("Training History:")
        st.write("Loss over epochs:", history.history['loss'])
        st.write("Validation Loss over epochs:", history.history['val_loss'])

elif uploaded_train is None or uploaded_test is None:
    st.sidebar.warning("Please upload both training and test data.")
