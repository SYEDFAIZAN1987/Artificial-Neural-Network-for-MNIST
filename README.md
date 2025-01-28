# 🧠 ANN for MNIST

## 🚀 Overview
This project implements an **Artificial Neural Network (ANN)** for classifying handwritten digits from the **MNIST dataset**. The model is built using **PyTorch** and utilizes **torchvision** for dataset preprocessing.

## ✨ Features
✅ **Built with PyTorch** - Leverages PyTorch for efficient model building and training.  
✅ **MNIST Dataset** - A widely used benchmark dataset for digit classification.  
✅ **Data Augmentation** - Applies `torchvision.transforms` for normalization and preprocessing.  
✅ **Performance Evaluation** - Includes accuracy measurement and confusion matrix visualization.  

## 📦 Installation
To set up the environment, install the required dependencies using:

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn
```

## 📊 Dataset
The MNIST dataset is automatically downloaded via `torchvision.datasets`. It consists of **60,000 training images** and **10,000 test images** of handwritten digits (0-9), each sized **28x28 pixels**.

## 🏗️ Model Architecture
The ANN model follows this architecture:
- 🔹 **Input Layer**: 784 neurons (flattened 28×28 image pixels)
- 🔹 **Hidden Layers**: Fully connected layers with **ReLU** activation
- 🔹 **Output Layer**: 10 neurons (representing digits 0-9) with **Softmax** activation

## 🏃‍♂️ Usage
### 🎯 Training the Model
Execute the following script to train the model:

```bash
python train.py
```

### 📈 Evaluating the Model
Once trained, you can evaluate its performance using:

```bash
python evaluate.py
```

## 📌 Results
The model performance is evaluated based on:
- 📍 **Accuracy**: Measures the percentage of correct predictions.
- 📍 **Confusion Matrix**: Visualizes misclassifications across digit classes.

### 📊 Example Output
```plaintext
🎯 Training Accuracy: 98.5%
✅ Validation Accuracy: 97.8%
📌 Confusion Matrix:
[[592   0   3   0   0   1   1   1   2   0]
 [  0 674   1   0   1   0   0   0   1   0]
 ...
```

## 🔗 References
📖 [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)  
📖 [Torchvision MNIST Dataset](https://pytorch.org/vision/stable/datasets.html#mnist)  

## 📜 License
📝 This project is licensed under the **MIT License**. Feel free to modify and use it!
