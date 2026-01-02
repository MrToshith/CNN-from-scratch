CNN From Scratch – MNIST Digit Recognition
==========================================

This project explores **Convolutional** Neural Networks from first principles and connects that understanding to a production-ready deep learning workflow for MNIST digit recognition. The goal is to understand how CNNs work internally while still delivering a high-accuracy, interactive application.

## Project Overview

The repository is divided into three logical parts.

1. CNN from Scratch (NumPy)  
   - Implemented convolution, ReLU, max-pooling, dense layer, and softmax manually.[1]
   - Focused on understanding data flow, tensor shapes, and learning behavior.  
   - Dense layer trained manually to demonstrate gradient-based learning.

2. TensorFlow CNN (Training)  
   - Proper CNN trained on MNIST using TensorFlow/Keras.[3]
   - Achieves approximately 97–98% accuracy.  
   - Model saved in modern `.keras` format.

3. Streamlit Visualization App  
   - Interactive canvas to draw digits in real time.[2]
   - Input resized and normalized to MNIST format.  
   - Uses the trained TensorFlow model for accurate predictions.[2]

## Tech Stack

- Python  
- NumPy  
- TensorFlow / Keras  
- Streamlit  
- PIL (image preprocessing)[4]

## Project Structure

```text
CNN-from-scratch/
├── notebooks/
│   ├── cnn_from_scratch_numpy.ipynb
│   └── cnn_training_tensorflow.ipynb
├── src/
│   └── app.py
├── models/
│   └── cnn_model.keras
└── README.md
```

## Key Learnings

- How convolution works as a sliding dot product.[5]
- Why pooling reduces spatial dimensions while preserving features.[1]
- Why CNNs require training to extract meaningful patterns.  
- How high-level frameworks optimize what is implemented manually.  
- How to deploy a trained model into an interactive application with Streamlit.[2]

## How to Run the App Locally

1. Install dependencies:
   ```bash
   pip install streamlit streamlit-drawable-canvas pillow tensorflow
   ```
2. Run Streamlit:
   ```bash
   streamlit run src/app.py
   ```

## Live Demo

- Live App: https://mrtoshith-cnn-from-scratch-srcapp-wuxy7c.streamlit.app/
