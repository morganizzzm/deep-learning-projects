Glaucoma Detection Project

Overview

This project aims to develop a neural network-based solution for detecting the presence of glaucoma in fundus images. Glaucoma is a severe eye condition that, if left undiagnosed and untreated, can lead to vision loss. Early detection is crucial for effective management and treatment.

The project is implemented using PyTorch and leverages transfer learning with a pre-trained ResNet-50 architecture. The model is trained on a dataset of fundus images labeled with two classes: 'Glaucoma Present' and 'Glaucoma not Present.' This README provides an overview of the project, including details about the neural network architecture used for glaucoma detection.

Neural Network Architecture

Model Overview
The neural network architecture employed for glaucoma detection is based on a modified ResNet-50 model. ResNet (Residual Network) is a widely-used deep convolutional neural network architecture known for its ability to train very deep networks effectively. Here's an overview of the architecture:

Feature Extraction Layers: The initial layers of the ResNet-50 model serve as a feature extractor. These layers are responsible for learning hierarchical features from the input fundus images. They include convolutional layers, batch normalization, ReLU activations, and max-pooling.
Fully Connected Layers: Following the feature extraction layers, we append custom fully connected layers to the model. These layers are responsible for learning the specific glaucoma detection features. The architecture typically consists of one or more fully connected layers followed by ReLU activation functions and dropout layers to reduce overfitting.
Output Layer: The final fully connected layer of the model has two output neurons, corresponding to the two classes: 'Glaucoma Present' and 'Glaucoma not Present.' A softmax activation function is applied to convert the model's output into class probabilities.
Transfer Learning
Transfer learning is employed by initializing the model with weights pre-trained on a large dataset (usually ImageNet) and fine-tuning it on the glaucoma detection dataset. This approach allows the model to leverage features learned from the broader dataset and adapt them to the specific task of glaucoma detection.

Model Training
The model is trained using a binary cross-entropy loss function, which is well-suited for binary classification tasks. AdamW optimizer is used with a learning rate scheduler to control the learning rate during training. The number of training epochs and batch size are hyperparameters that can be adjusted based on your dataset and computational resources.

Usage


