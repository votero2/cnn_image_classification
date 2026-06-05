# CNN Letter, Digit, and Symbol Recognition System

## Overview

This project implements a Convolutional Neural Network (CNN) based Optical Character Recognition (OCR) system capable of recognizing handwritten or printed characters from images.

The system is designed to classify:

* Digits (0-9)
* Letters (A-Z)
* Selected Symbols (such as $, %, &, @, etc.)

The project combines image preprocessing, deep learning, and computer vision techniques to provide accurate character recognition from user-supplied images.

---

## Features

* Character image preprocessing
* CNN-based digit recognition
* CNN-based letter recognition
* Symbol recognition support
* Image segmentation
* Character prediction and display
* TensorFlow/Keras implementation
* Python-based solution

---

## Technologies Used

* Python 3.x
* TensorFlow
* Keras
* OpenCV
* NumPy
* Matplotlib
* Pillow (PIL)

---

## Project Architecture

Image Input
↓
Image Preprocessing
↓
Character Segmentation
↓
Resize and Normalize
↓
CNN Model
↓
Prediction
↓
Character Output

---

## Dataset

The model was trained using labeled images representing:

### Digits

0, 1, 2, 3, 4, 5, 6, 7, 8, 9

### Letters

A-Z

### Symbols

Examples:

* $
* %
* &
* @
* #
* !

The dataset is organized into class folders where each folder represents a character label.

Example:

dataset/

├── digits/

│ ├── 0/

│ ├── 1/

│ └── ...

├── letters/

│ ├── A/

│ ├── B/

│ └── ...

└── symbols/

├── dollar/

├── percent/

└── ...

---

## Image Preprocessing

Before classification, images undergo several preprocessing steps:

1. Grayscale conversion
2. Noise reduction
3. Thresholding
4. Character extraction
5. Resizing
6. Normalization

These steps improve recognition accuracy and reduce variations in handwriting or image quality.

---

## CNN Architecture

The Convolutional Neural Network consists of:

* Convolutional Layers
* ReLU Activation Functions
* Max Pooling Layers
* Flatten Layer
* Dense Layers
* Softmax Output Layer

Example architecture:

Input Image (28x28)
↓
Conv2D
↓
ReLU
↓
MaxPooling
↓
Conv2D
↓
ReLU
↓
MaxPooling
↓
Flatten
↓
Dense
↓
Dropout
↓
Dense (Output)
↓
Softmax

---

## Training Configuration

Typical training settings:

* Optimizer: Adam
* Loss Function: Sparse Categorical Crossentropy
* Epochs: 20-50
* Batch Size: 32
* Activation Function: ReLU
* Output Activation: Softmax

---

## Model Training

Train the model using:

```bash
python train_model.py
```

The training process:

1. Loads images
2. Splits data into training and validation sets
3. Trains the CNN
4. Saves the model weights
5. Displays accuracy and loss graphs

````

## Running Predictions

To predict characters from an image:

```bash
python predict.py
````

The system will:

1. Load the image
2. Preprocess the image
3. Extract characters
4. Predict each character
5. Display the results

````

## Example Predictions

Input:

```text
A
````

Prediction:

```text
A
```

Input:

```text
7
```

Prediction:

```text
7
```

Input:

```text
$
```

Prediction:

```text
$
```

---

## Results

The model successfully recognizes:

* Digits
* Alphabetic characters
* Common symbols

Performance depends on:

* Dataset size
* Image quality
* Character clarity
* Training duration

---

## Challenges Encountered

During development several challenges were addressed:

* Incorrect image resizing
* Class imbalance
* Symbol recognition difficulties
* Dataset preprocessing issues
* Character segmentation improvements
* Model overfitting prevention

Solutions included:

* Improved normalization
* Additional training samples
* Better image preprocessing
* CNN architecture adjustments

---

## Future Improvements

Planned enhancements include:

* Larger training datasets
* Additional symbol support
* Handwriting recognition improvements
* Real-time webcam recognition
* Transformer-based OCR models
* GUI application integration
* Multi-character word recognition

---

## Educational Objectives

This project demonstrates:

* Convolutional Neural Networks (CNNs)
* Computer Vision
* Optical Character Recognition (OCR)
* Deep Learning Model Training
* Image Processing
* TensorFlow/Keras Development

---

## Author

Developed as part of Artificial Intelligence and Deep Learning coursework to explore computer vision and character recognition using Convolutional Neural Networks.
