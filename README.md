# CNN-Based Image Classification using CIFAR-10

## Overview

This project implements a **Convolutional Neural Network (CNN)** for image classification using the **CIFAR-10 dataset**.
The model is designed and trained from scratch to learn hierarchical visual features such as edges, textures, and object shapes for accurate classification.

The goal of this project is to build and train a custom CNN architecture capable of classifying images into ten different object categories.

---

## Dataset

The **CIFAR-10 dataset** is a widely used benchmark dataset for image classification.

Dataset characteristics:

* **60,000 color images**
* **10 object classes**
* **Image size:** 32 × 32 pixels
* **50,000 training images**
* **10,000 testing images**

Classes in the dataset:

* Airplane
* Automobile
* Bird
* Cat
* Deer
* Dog
* Frog
* Horse
* Ship
* Truck

---

## Model Architecture

The CNN architecture consists of three convolutional blocks followed by fully connected layers.

### Convolution Block 1

* Conv2D (32 filters, 3×3 kernel)
* Batch Normalization
* LeakyReLU Activation
* MaxPooling (2×2)

### Convolution Block 2

* Conv2D (64 filters, 3×3 kernel)
* Batch Normalization
* LeakyReLU Activation
* MaxPooling (2×2)

### Convolution Block 3

* Conv2D (128 filters, 3×3 kernel)
* Batch Normalization
* LeakyReLU Activation
* MaxPooling (2×2)

### Fully Connected Layers

* Flatten Layer
* Dense Layer (256 neurons)
* LeakyReLU Activation
* Dropout (0.3)
* Output Layer (Softmax)

The Softmax layer outputs probability scores for the 10 CIFAR-10 classes.

---

## Training Configuration

Model training parameters:

* **Optimizer:** Adam
* **Learning Rate:** 0.0005
* **Loss Function:** Categorical Crossentropy
* **Batch Size:** 32
* **Epochs:** 10
* **Evaluation Metric:** Accuracy

---

## Training Process

The workflow for training the CNN model:

1. Load the CIFAR-10 dataset
2. Normalize image pixel values
3. Convert labels into categorical format
4. Build the CNN architecture
5. Compile the model
6. Train the model using the training dataset
7. Evaluate performance on the test dataset

---

## Results

After training, the model achieves approximately:

**Test Accuracy: ~80%**

This demonstrates the effectiveness of convolutional neural networks for image classification tasks on small-scale datasets.

---

## Project Structure

```
CNN-CIFAR10-Classification
│
├── CNN_CIFAR10.ipynb
├── requirements.txt
├── README.md
└── results
    └── training_accuracy.png
```

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/CNN-CIFAR10-Classification.git
cd CNN-CIFAR10-Classification
pip install -r requirements.txt
```

---

## Running the Project

Open the notebook and run all cells:

```
CNN_CIFAR10.ipynb
```

The notebook will:

1. Load the dataset
2. Train the CNN model
3. Evaluate performance
4. Display classification accuracy

---

## Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib
* Jupyter Notebook

---

## Applications

CNN-based image classification models like this are used in:

* Object detection systems
* Autonomous driving
* Medical image analysis
* Surveillance systems
* Image search engines

---

## Author

Computer Vision Laboratory

Aparajita Vaish
253100101
Mtech ECE 

---

## License

This project is for academic and educational purposes.
