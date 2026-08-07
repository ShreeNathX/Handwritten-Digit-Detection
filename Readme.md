# Handwritten Digit Recognition using CNN

A deep learning project that recognizes handwritten digits (0–9) using a Convolutional Neural Network (CNN) trained on the MNIST dataset. The application provides an interactive drawing canvas where users can write digits and receive real-time predictions with confidence scores and probability distributions.

---

## Live Demo

**Streamlit App:**  
https://YOUR-STREAMLIT-LINK.streamlit.

---

## Project Overview

This project demonstrates how Convolutional Neural Networks (CNNs) can accurately classify handwritten digits. The model is trained on the MNIST dataset and integrated into a Streamlit web application that allows users to draw digits directly in the browser.

To improve real-world performance, the application preprocesses the drawn digit by cropping, resizing, and centering it before prediction, closely matching the format of the original MNIST images.

---

## Features

- Interactive drawing canvas
- Real-time handwritten digit recognition
- Prediction confidence score
- Probability distribution for all 10 digit classes
- Automatic image preprocessing
- CNN model trained using TensorFlow/Keras
- User-friendly Streamlit interface

---

## Tech Stack

### Programming Language
- Python

### Deep Learning
- TensorFlow
- Keras

### Web Framework
- Streamlit
- streamlit-drawable-canvas

### Data Processing
- NumPy
- Pillow

### Data Visualization
- Matplotlib
- Seaborn

### Machine Learning Utilities
- Scikit-learn

---

## Dataset

The project uses the **MNIST Handwritten Digit Dataset**, containing:

- **70,000** grayscale images
- **60,000** training images
- **10,000** testing images
- Image size: **28 × 28 pixels**
- Digit classes: **0–9**

MNIST is one of the most widely used benchmark datasets for image classification.

---

## Model Architecture

The model is a Convolutional Neural Network (CNN) consisting of:

- Convolutional Layers
- ReLU Activation
- Max Pooling Layers
- Dropout Layers
- Fully Connected Dense Layers
- Softmax Output Layer

Data augmentation techniques such as rotation, shifting, and zooming were applied during training to improve generalization.

---

## Project Structure

```
digit-recognizer/
│
├── train_notebook.ipynb     # Complete training pipeline
├── app.py                   # Streamlit application
├── digit_model.keras        # Trained CNN model
├── requirements.txt         # Required Python libraries
├── README.md
```

---

## Workflow

1. Load the MNIST dataset.
2. Perform preprocessing and normalization.
3. Apply data augmentation.
4. Train the CNN model.
5. Evaluate the model.
6. Save the trained model.
7. Launch the Streamlit application.
8. Draw a digit on the canvas.
9. Preprocess the drawing.
10. Predict the digit with confidence.

---

## Model Performance

| Metric | Value |
|---------|--------|
| Test Accuracy | **99.33%** |
| Dataset | MNIST |
| Classes | 10 |
| Image Size | 28 × 28 |

### Performance Highlights

- **Test Accuracy:** **99.33%**
- High confidence predictions for clean handwritten digits
- Robust preprocessing significantly improves recognition of user-drawn digits
- Displays class-wise probability distribution for every prediction

---

## Image Preprocessing

Before prediction, every drawing undergoes:

- Conversion to grayscale
- Bounding box detection
- Cropping the handwritten digit
- Resizing to a 20 × 20 region
- Centering inside a 28 × 28 image
- Pixel normalization

This preprocessing closely matches the original MNIST format and improves prediction accuracy.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/digit-recognizer.git

cd digit-recognizer
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Train the Model

Run the notebook:

```bash
train_notebook.ipynb
```

The notebook will:

- Download MNIST
- Train the CNN
- Evaluate performance
- Save the trained model as:

```
digit_model.keras
```

---

## Run the Application

```bash
streamlit run app.py
```

Open the local Streamlit URL in your browser.

Draw a digit and click **Predict**.

---

## Future Improvements

- Multi-digit recognition
- Support for handwritten mathematical expressions
- Model quantization for faster inference
- Mobile-friendly interface
- Deploy using Docker
- Support EMNIST alphabet recognition

---

## License

This project is licensed under the **MIT License**.

---

## Acknowledgements

- TensorFlow
- Streamlit
- MNIST Dataset
- Keras
- Scikit-learn

---

## Author

**Shree Nath Mahato**

---