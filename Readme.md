# Handwritten Digit Recognition using CNN

A deep learning project that recognizes handwritten digits (0–9) using a Convolutional Neural Network (CNN) trained on the MNIST dataset. The application provides an interactive drawing canvas where users can write digits and receive real-time predictions with confidence scores and probability distributions.

---

## Live Demo

**Streamlit App:**  
https://handwritten-digit-detection.streamlit.app/

---

## Project Overview

This project demonstrates handwritten digit recognition using a CNN trained on the MNIST dataset. Users can draw digits directly on an interactive canvas, and the application predicts the digit in real time after applying preprocessing techniques such as cropping, centering, resizing, and normalization.

---

## Features

- Interactive drawing canvas
- Real-time handwritten digit prediction
- Prediction confidence score
- Class-wise probability distribution
- Automatic image preprocessing
- CNN model trained on the MNIST dataset
- Simple and responsive Streamlit interface

---

## Tech Stack

- Python **3.11**
- TensorFlow / Keras
- Streamlit
- streamlit-drawable-canvas
- NumPy
- Pillow
- Matplotlib
- Seaborn
- Scikit-learn

---

## Dataset

The project uses the **MNIST Handwritten Digit Dataset**, which contains:

- 70,000 grayscale handwritten digit images
- 60,000 training images
- 10,000 testing images
- Image size: **28 × 28 pixels**
- Digit classes: **0–9**

---

## Model Architecture

The model is a Convolutional Neural Network (CNN) consisting of:

- Convolutional Layers
- ReLU Activation
- Max Pooling Layers
- Dropout Layers
- Dense Layers
- Softmax Output Layer

Data augmentation techniques such as rotation, shifting, and zooming are applied during training to improve generalization.

---

## Project Structure

```text
handwritten-digit-detection/
│
├── train_notebook.ipynb
├── app.py
├── digit_model.keras
├── requirements.txt
├── runtime.txt
├── README.md
```

---

## Workflow

1. Load the MNIST dataset.
2. Preprocess and normalize the images.
3. Apply data augmentation.
4. Train the CNN model.
5. Evaluate the model.
6. Save the trained model.
7. Launch the Streamlit application.
8. Draw a digit and predict the result.

---

## Model Performance

| Metric | Value |
|---------|------:|
| Test Accuracy | **99.33%** |
| Dataset | MNIST |
| Number of Classes | 10 |
| Image Size | 28 × 28 |

### Performance Highlights

- **99.33% Test Accuracy**
- High-confidence predictions for handwritten digits
- Image preprocessing improves prediction accuracy on user-drawn digits
- Displays probability scores for all digit classes

---

## Image Preprocessing

Before prediction, each drawing undergoes:

- Grayscale conversion
- Bounding box detection
- Cropping
- Resizing to 20 × 20 pixels
- Centering in a 28 × 28 frame
- Pixel normalization

---

## Installation

Clone the repository:

```bash
git clone https://github.com/ShreeNathX/Handwritten-Digit-Detection.git
cd handwritten-digit-detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Python Version

This project is developed and tested using:

```text
Python 3.11
```

> **Note:** TensorFlow currently does not support Python 3.14 on Streamlit Cloud. Using **Python 3.11** is recommended for successful deployment.

---

## Train the Model

Run the notebook:

```bash
train_notebook.ipynb
```

The notebook will:

- Download the MNIST dataset
- Train the CNN model
- Evaluate model performance
- Save the trained model as:

```text
digit_model.keras
```

---

## Run the Application

```bash
streamlit run app.py
```

Open the local Streamlit URL in your browser, draw a digit, and click **Predict**.

---

## Future Improvements

- Multi-digit recognition
- Handwritten mathematical expression recognition
- Faster inference using model optimization
- Mobile-friendly interface
- Docker support
- EMNIST alphabet recognition

---

## License

This project is licensed under the **MIT License**.

---

## Author

**Shree Nath Mahto**