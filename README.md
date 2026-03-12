# CNN for CIFAR-10 Image Classification

## Computer Vision Lab Assignment

This repository implements and trains a **Convolutional Neural Network (CNN)** for image classification using the CIFAR-10 dataset.

---

# Project Overview

The objective of this project is to design and train a custom CNN architecture capable of classifying images from the CIFAR-10 dataset into ten different categories.

The project demonstrates key computer vision concepts including:

* Image preprocessing
* Convolutional Neural Networks
* Model training and evaluation
* Performance analysis

---

# Dataset

The project uses the **CIFAR-10 dataset**.

Dataset characteristics:

* 60,000 color images
* Image resolution: 32 × 32 pixels
* 10 image classes
* 50,000 training images
* 10,000 testing images

Classes included in the dataset:

* airplane
* automobile
* bird
* cat
* deer
* dog
* frog
* horse
* ship
* truck

Dataset source:
https://www.cs.toronto.edu/~kriz/cifar.html

---

# CNN Architecture

The model is built using three convolutional blocks followed by fully connected layers.

## Convolution Block 1

* Conv2D (32 filters, 3×3 kernel)
* Batch Normalization
* LeakyReLU Activation
* MaxPooling (2×2)

## Convolution Block 2

* Conv2D (64 filters, 3×3 kernel)
* Batch Normalization
* LeakyReLU Activation
* MaxPooling (2×2)

## Convolution Block 3

* Conv2D (128 filters, 3×3 kernel)
* Batch Normalization
* LeakyReLU Activation
* MaxPooling (2×2)

## Fully Connected Layers

* Flatten
* Dense Layer (256 neurons)
* LeakyReLU Activation
* Dropout (0.3)

## Output Layer

* Dense Layer (10 neurons)
* Softmax Activation

---

# Training Configuration

| Parameter     | Value                    |
| ------------- | ------------------------ |
| Optimizer     | Adam                     |
| Loss Function | Categorical Crossentropy |
| Batch Size    | 32                       |
| Epochs        | 10                       |

---

# Repository Structure

```
cnn-cifar10-computer-vision-lab
│
├── cnn.ipynb
│
├── src
│   └── train_cnn.py
│
├── results
│   └── training_output.png
│
├── requirements.txt
│
├── README.md
│
├── .gitignore
│
└── LICENSE
```

---

# Installation

Clone the repository

```
git clone https://github.com/YOUR_USERNAME/cnn-cifar10-computer-vision-lab.git
```

Navigate to project directory

```
cd cnn-cifar10-computer-vision-lab
```

Install required dependencies

```
pip install -r requirements.txt
```

---

# Run Training Script

```
python src/train_cnn.py
```

---

<!----# Model Training Code

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

NUM_CLASSES = 10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)

input_layer = layers.Input((32,32,3))

x = layers.Conv2D(32,(3,3),padding="same")(input_layer)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)
x = layers.MaxPooling2D((2,2))(x)

x = layers.Conv2D(64,(3,3),padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)
x = layers.MaxPooling2D((2,2))(x)

x = layers.Conv2D(128,(3,3),padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)
x = layers.MaxPooling2D((2,2))(x)

x = layers.Flatten()(x)
x = layers.Dense(256)(x)
x = layers.LeakyReLU()(x)
x = layers.Dropout(0.3)(x)

output_layer = layers.Dense(NUM_CLASSES,activation="softmax")(x)

model = models.Model(input_layer,output_layer)

model.compile(
optimizer="adam",
loss="categorical_crossentropy",
metrics=["accuracy"]
)

model.fit(
x_train,
y_train,
batch_size=32,
epochs=10,
validation_data=(x_test,y_test)
)

test_loss, test_acc = model.evaluate(x_test,y_test)

print("Test Accuracy:", test_acc)
```---->

---

# Results

Expected model performance:

Test Accuracy: **~78% – 82%**

The model successfully learns meaningful features from the CIFAR-10 dataset using convolutional layers and achieves good classification accuracy for a basic CNN architecture.

---

# Author

Name: Aparajita Vaish
RollNo: 253100101
Computer Vision Lab
Mtech ECE
Dr. Shyama Prasad Mukherjee International Institute of Information Technology, Naya Raipur

---

# License

This project is licensed under the **MIT License**.
