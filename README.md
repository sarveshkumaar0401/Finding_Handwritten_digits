# 🔢 Handwritten Digit Recognition with Neural Networks

This project implements a fully connected neural network (Multi-Layer Perceptron) using TensorFlow and Keras to classify handwritten digits from the **MNIST** dataset.

---

## 📊 Dataset

- **MNIST** is a classic dataset containing 70,000 grayscale images of handwritten digits (0–9).
- Images are 28×28 pixels.
- Preloaded via `tensorflow.keras.datasets.mnist`.

---

## 🧠 Model Architecture

- **Input Layer**: 784 units (flattened 28×28 image)
- **Hidden Layer 1**: Dense(128), ReLU activation  
- **Hidden Layer 2**: Dense(64), ReLU activation  
- **Output Layer**: Dense(10), Softmax (for 10-digit classification)

---

## ⚙️ Preprocessing

- Pixel values normalized to range `[0, 1]`
- Labels converted to one-hot vectors using `to_categorical`

---

## 🏃‍♂️ Training

```python
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)
