# 🧠 ANN for MNIST

## 🔄 Sequence Diagram
This sequence diagram illustrates the flow of data from dataset loading to training and evaluation.

![Sequence Diagram](uml%20ann.png)

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
[[   0    1    2    3    4    5    6    7    8    9]]

[[ 968    0    1    0    0    1    6    1    1    1]
 [   1 1122    1    0    0    0    2    3    0    2]
 [   0    1 1010    1    3    0    2    5    6    0]
 [   1    4    9  997    0   26    1    4   11    8]
 [   1    0    2    0  966    2    8    3    5   11]
 [   1    0    0    2    1  851    5    0    2    4]
 [   1    2    0    0    3    3  926    1    1    0]
 [   1    1    2    3    2    1    0  999    2    0]
 [   5    5    7    5    1    4    8    4  943    5]
 [   1    0    0    2    6    4    0    8    3  978]]
```
### 📊 Accuracy Plot
![Training and Validation Accuracy](Accuracy%20of%20the%20training%20and%20test%20data.png)

### 📉 Loss Plot
![Training and Validation Loss](Loss%20after%20each%20Epoch.png)



## 🔗 References
📖 [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)  
📖 [Torchvision MNIST Dataset](https://pytorch.org/vision/stable/datasets.html#mnist)  

## 📜 License
📝 This project is licensed under the **MIT License**. Feel free to modify and use it!
