import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image

# Define the path to the root folder containing subfolders for each digit class
data_dir = 'data/trainingSet/trainingSet/'

# Initialize empty lists to store images and their corresponding labels
images = []
labels = []

# Loop through each digit class folder
for digit_class in range(10):
    class_dir = os.path.join(data_dir, str(digit_class))
    
    # Loop through each image file in the class folder
    for filename in os.listdir(class_dir):
        if filename.endswith('.jpg'):
            # Load the image using PIL
            img = Image.open(os.path.join(class_dir, filename))
            
            # Convert the image to grayscale (if needed)
            img = img.convert('L')
            
            # Normalize the pixel values to the range [0, 1]
            img = np.array(img, dtype=np.float32) / 255.0
            
            # Append the image to the images list
            images.append(img)
            
            # Append the corresponding label (digit class) to the labels list
            labels.append(digit_class)

# Convert the lists to NumPy arrays
images = np.array(images)
labels = np.array(labels)

# Shuffle the data (optional)
indices = np.arange(len(images))
np.random.shuffle(indices)
images = images[indices]
labels = labels[indices]

# Split the data into training, validation, and test sets
split_ratio = 0.7  # Adjust this ratio as needed
split_idx = int(len(images) * split_ratio)
train_images, val_test_images = images[:split_idx], images[split_idx:]
train_labels, val_test_labels = labels[:split_idx], labels[split_idx:]
val_images, test_images = val_test_images[:len(val_test_images) // 2], val_test_images[len(val_test_images) // 2:]
val_labels, test_labels = val_test_labels[:len(val_test_labels) // 2], val_test_labels[len(val_test_labels) // 2:]

# Define the CNN model
model = models.Sequential([
    # Convolutional layers
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Fully connected layers
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')  # Output layer with 10 units for 0-9 digit classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10, validation_data=(val_images, val_labels))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test accuracy:", test_acc)

# Save the trained model (optional)
model.save('digit_recognition_model.h5')
