# MNIST Handwritten Digit Classification

This repository contains **two separate classifiers** built on the classic [MNIST handwritten digits dataset](https://www.kaggle.com/competitions/digit-recognizer).
![](/MNIST_dataset_example.png)

The goal is to recognize digits (0–9) from 28×28 grayscale images using two approaches:

1. **From Scratch Neural Network** — built entirely using **NumPy**, implementing all steps manually (forward propagation, backpropagation, gradient descent, etc.).
2. **CNN Model with Data Augmentation** — built using **Keras/TensorFlow**, demonstrating a modern convolutional neural network approach with image preprocessing and augmentation.

---

## 📊 Dataset

The [MNIST dataset](https://www.kaggle.com/competitions/digit-recognizer/data) contains 70,000 grayscale images of handwritten digits (0–9), each of size **28x28 pixels**.  
- **Training set:** 60,000 images  
- **Test set:** 10,000 images  

Both implementations in this repo use the Kaggle “Digit Recognizer” version of the dataset:
- `train.csv` (42,000 samples)
- `test.csv` (28,000 samples)

---

## MNIST Classifier — From Scratch

### Overview
This project builds a simple **2-layer fully connected neural network** using only NumPy — no deep learning libraries. It helps understand how neural networks work internally.

### Key Concepts Implemented
- One-hot encoding  
- Forward propagation  
- ReLU and Softmax activations  
- Backward propagation (manual gradient computation)  
- Parameter updates with gradient descent  

---

## MNIST Classifier — CNN with Data Augmentation

### Overview
A **Convolutional Neural Network (CNN)** implemented using **TensorFlow and Keras** for digit recognition. Data augmentation is applied to improve generalization.

## Performance Comparison

 | Criteria                       | From Scratch NN                  | CNN (Keras)                       |
| ---------------------------- | -------------------------------- | --------------------------------- |
| Framework                    | NumPy                            | Keras / TensorFlow                |
| Model Type                   | Fully Connected  | Deep Convolutional Neural Network |
| Data Augmentation            | ❌ No                             | ✅ Yes                             |
| Training Iterations / Epochs | 1200                             | 15                                |
| Optimizer                    | Manual Gradient Descent          | RMSprop                           |
| Training Accuracy        | **99%**                        | **99.12%**            |
| Validation Accuracy        | **96.553%**                        | **99.103%** |
| Overfitting                  | Moderate                         | Minimal                           |