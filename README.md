# MNIST-Digit-Classification
This project demonstrates how to build a deep learning model to classify handwritten digits from the MNIST dataset. It covers data loading, preprocessing, model architecture, training, evaluation, and a prediction system for classifying user-provided images.

This project demonstrates how to build a deep learning model to classify handwritten digits from the MNIST dataset.

## Dependencies

* NumPy
* Matplotlib
* TensorFlow
* Seaborn
* OpenCV
* Keras
* PIL

## Dataset

The MNIST dataset is a large collection of handwritten digits (0-9) commonly used for training and evaluating image classification models. It contains 60,000 training images and 10,000 testing images, each of size 28x28 pixels and in grayscale.

## Model Architecture

The neural network model consists of the following layers:

* Flatten layer to convert the 28x28 input images into a 1D vector.
* Dense layer with 512 neurons and ReLU activation.
* Dense layer with 256 neurons and ReLU activation.
* Dense layer with 100 neurons and Sigmoid activation.
* Output dense layer with 10 neurons (one for each digit) and Sigmoid activation.

## Training

The model is trained using the Adam optimizer and sparse categorical crossentropy loss function. It is trained for 10 epochs on the MNIST training data.

## Evaluation

The model achieves an accuracy of 99.4% on the training data and 97.4% on the testing data. A confusion matrix is generated to visualize the model's performance on each digit class.

## Prediction System

A prediction system is implemented to classify handwritten digits from user-provided images. The system takes an image path as input, preprocesses the image (resizing, grayscale conversion, normalization), and feeds it to the trained model for prediction. The predicted digit label is then displayed.

## Usage

1. Install the required dependencies.
2. Run the Jupyter notebook to train the model and evaluate its performance.
3. Use the prediction system to classify handwritten digits from your own images.
