import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model('digit_recognition_model.h5')  # Change the filename if needed

# Function to preprocess a single image for prediction
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.convert('L')  # Convert to grayscale
    img = np.array(img, dtype=np.float32) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to predict the digit from an image
def predict_digit(image_path):
    # Preprocess the image
    img = preprocess_image(image_path)
    
    # Make the prediction
    predictions = model.predict(img)
    
    # Get the predicted digit (class)
    predicted_digit = np.argmax(predictions)
    
    return predicted_digit

# Example usage
image_path = 'data/testSample/img_2.jpg'  # Change to the path of your handwritten digit image
predicted_digit = predict_digit(image_path)
print(f'Predicted Digit: {predicted_digit}')
