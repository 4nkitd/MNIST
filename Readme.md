
# Handwritten Digit Recognition with Deep Learning

This project demonstrates the use of a deep learning model to recognize handwritten digits (0-9) using TensorFlow and Keras. The model is trained on a dataset of handwritten digit images and can make predictions on new images of handwritten digits.

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine for testing and development purposes.

### Prerequisites

To run the code in this project, you need the following dependencies installed on your system:

- Python 3.x
- TensorFlow 2.x
- NumPy
- Pillow (PIL)

#### Data to download

```urlhttps://github.com/4nkitd/MNIST
https://www.kaggle.com/datasets/scolianni/mnistasjpg/
```

You can install these dependencies using the following commands:

```bash
pip install tensorflow numpy pillow
```

### Dataset

The dataset used for training this model should be organized as follows:

```
data/
    trainingSet/
        0/
            img_0.jpg
            img_1.jpg
            ...
        1/
            img_0.jpg
            img_1.jpg
            ...
        ...
        9/
            img_0.jpg
            img_1.jpg
            ...
```

Each digit class (0-9) has its own subfolder, and images are named as `img_{num}.jpg`.

### Training the Model

1. Organize your dataset as described above.
2. Run the training script to train the model:

```bash
python trainer.py
```

3. The trained model will be saved as `'digit_recognition_model.h5'`.

### Using the Trained Model

You can use the trained model to make predictions on new handwritten digit images. Create a Python script and load the model as shown in `predict_digit.py`:

```python
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model('digit_recognition_model.h5')

# Preprocess an image and predict the digit
# Replace 'sample_digit.jpg' with the path to your own image
image_path = 'sample_digit.jpg'
predicted_digit = predict_digit(image_path)
print(f'Predicted Digit: {predicted_digit}')
```

Replace `'sample_digit.jpg'` with the path to your handwritten digit image.

