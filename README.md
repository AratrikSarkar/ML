# 🧠 Different Types of MODELS  

> A collection of deep learning models and core components implemented from scratch using PyTorch.  
> This repository is designed for **learning and understanding the inner workings** of common architectures like CNNs, RNNs (GRU, LSTM), and basic layers, without relying on pre-built PyTorch modules.  

---

## 📌 Table of Contents  
- [About](#about)    
- [Features](#features)  
- [Implemented Models](#implemented-models)  
- [Installation](#installation)  
- [Usage](#usage)  
- [File Descriptions](#file-descriptions)  
- [Future Work](#future-work)  

---

## 📖 About  
This project provides **from-scratch implementations** of popular deep learning models and components in PyTorch.  
Instead of using high-level APIs like `torch.nn.Conv2d` or `torch.nn.LSTM`, the repository manually defines convolution, recurrent units, activation functions, dropout, and more.  

This allows learners to:  
- Understand **how neural networks actually compute** at a low level.  
- Experiment with **DNA sequence data (AeCa.txt)** to test sequence models.  
- Gain insight into **forward propagation, parameter initialization, and layer mechanics**.  

---

## ✨ Features  
- ✅ **From-Scratch Implementations** (Conv2D, GRU, LSTM, etc.)  
- ✅ **Custom Utility Functions** (Softmax, Dropout, Flatten)  
- ✅ **Educational Jupyter Notebook** for step-by-step walkthrough  
- ✅ **DNA Sequence Dataset** for character-level modeling  
- ✅ **Lightweight & Modular** code to plug and play  

---

## 🏗 Implemented Models  

1. **CNN (Convolutional Neural Network)** – manual implementation of 2D convolutions.  
2. **GRU (Gated Recurrent Unit)** – step-by-step gates and hidden state update.  
3. **LSTM (Long Short-Term Memory)** – implementation of input, forget, output gates, and cell state.  
4. **Linear Layers & Activations** – including custom `Linear`, `ReLU`, `Sigmoid`, and `Tanh`.  
5. **Utility Layers** – Dropout, Softmax, Flatten, etc.  

---

## ⚙️ Installation  

Clone the repository:  

```bash
git clone https://github.com/your-username/Different-Types-of-MODELS.git
cd Different-Types-of-MODELS
```

Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
```
Install dependencies:

```bash
pip install torch jupyter
```

## 🚀 Usage

Run Jupyter Notebook:

```bash
jupyter notebook JustSomeCode.ipynb
```

Example: Using the CNN Module

```bash
from CNN import Conv2D
import torch

# Input: batch of 1, 3 channels, 32x32 image
x = torch.randn(1, 3, 32, 32)
conv = Conv2D(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
y = conv(x)

print(y.shape)  # -> (1, 8, 32, 32)
```

Example: Encoding DNA Sequences

```bash
with open("AeCa.txt", "r", encoding="UTF-8") as f:
    text = f.read()

chars = sorted(list(set(text)))   # ['A', 'C', 'G', 'T']
stoi = {s: i for i, s in enumerate(chars)}
itos = {i: s for s, i in stoi.items()}

encode = lambda seq: [stoi[s] for s in seq]
decode = lambda nums: ''.join([itos[i] for i in nums])

print(encode("GTTA"))  # -> [2, 3, 3, 0]
print(decode([2, 3, 3, 0]))  # -> GTTA
```

## 📂 File Descriptions  

- **AeCa.txt** → Dataset containing DNA sequences (**A, C, G, T**).  
- **CNN.py** → Custom 2D convolution layer (`Conv2D`) with forward pass written manually.  
- **Functions.py** → Implements utility functions like **Softmax, Dropout, and Flatten**.  
- **GRU.py** → Defines **Gated Recurrent Unit (GRU)** architecture using custom gates.  
- **LSTM.py** → Manual implementation of **LSTM architecture** with cell state management.  
- **Model1.py** → Core neural network layers such as **Linear, ReLU, Sigmoid, Tanh** implemented manually.  
- **JustSomeCode.ipynb** → Demonstrates **dataset preprocessing, encoding/decoding, and training samples**.  

## 🔮 Future Work

- Add **training pipelines** for CNN/GRU/LSTM.
- Include **visualizations** of hidden states and feature maps.
- Expand dataset support beyond DNA sequences.
- Implement **attention mechanisms** from scratch.

