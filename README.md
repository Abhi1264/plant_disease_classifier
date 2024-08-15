# Leaf-Guard: AI-Powered Plant Disease Classifier

## Overview

Leaf-Guard is an AI-powered classifier designed to determine whether the leaf of a plant is diseased or healthy. This project leverages transfer learning with a pre-trained neural network to classify plant leaves into various disease categories. The model is trained on the PlantVillage dataset and can be used through a web application built with Flask.

## Features

- **Leaf Disease Detection**: Classifies plant leaves as healthy or diseased based on images.
- **Web Interface**: Upload images and get predictions through a user-friendly web application.
- **Transfer Learning**: Utilizes the EfficientNetB0 model pre-trained on ImageNet to improve performance.

## Dataset

The model is trained using the PlantVillage dataset, which contains images of plant leaves categorized into 38 classes representing different diseases and healthy conditions.

## Neural Network Architecture

### Base Model

- **EfficientNetB0**: A state-of-the-art convolutional neural network architecture that balances model accuracy with computational efficiency. EfficientNetB0 is used as the base model due to its excellent performance on image classification tasks.

### Custom Layers

- **GlobalAveragePooling2D**: Reduces the spatial dimensions of the feature maps to a single vector, allowing the model to focus on important features.
- **Dense Layers**:
  - **1st Layer**: A fully connected 64 unit layer with ReLU activation to introduce non-linearity.
  - **2nd Layer**: The output layer with softmax activation for multi-class classification.

### Model Training

- **Optimizer**: Adam optimizer is used for model training, known for its efficiency in handling large datasets.
- **Loss Function**: Categorical crossentropy, suitable for multi-class classification problems.

The model is trained  with early stopping to prevent overfitting. The `EarlyStopping` callback monitors the validation loss and restores the best weights when there is no improvement.

## Image Preprocessing

- **Rescaling**: Images are rescaled to normalize pixel values between 0 and 1.
- **Augmentation**: During training, image data is augmented using techniques such as rotation, width and height shifts, shear, zoom, and horizontal flips to improve model generalization.

## Web Application

The web application is built using Flask and allows users to upload an image of a plant leaf to get predictions. The application has the following routes:

- **`/`**: The home page, where users can upload images.
- **`/predict`**: Handles image uploads, processes the image, and returns the prediction result.
- **`/uploads/<filename>`**: Serves the uploaded images for display.

### Running the Flask App

1. **Setup Environment**:
   Ensure you have the necessary libraries installed:
   ```bash
   pip install tensorflow flask numpy matplotlib
2. **Run the Flask App**:
   Start the flask app
   ```bash
   python -m flask run

## Accuracy Metrics

- **Validation Accuracy**: 0.9322
- **Precision**:
  - **Macro Average**: 0.92
  - **Weighted Average**: 0.94
- **Recall**:
  - **Macro Average**: 0.91
  - **Weighted Average**: 0.93
- **F1 Score**:
  - **Macro Average**: 0.90
  - **Weighted Average**: 0.93
