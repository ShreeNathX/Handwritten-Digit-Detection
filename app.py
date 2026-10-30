import logging
from pathlib import Path

import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow import keras
from PIL import Image

LOGGER = logging.getLogger(__name__)
MODEL_PATH = Path(__file__).resolve().with_name("digit_model.keras")
NUM_CLASSES = 10

st.title("Handwritten Digit Recognition")


@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    # The model is used only for inference. Keep Keras safe-mode enabled so
    # serialized Lambda layers cannot execute arbitrary Python code.
    return keras.models.load_model(MODEL_PATH, compile=False, safe_mode=True)


try:
    model = load_model()
except Exception:
    LOGGER.exception("Could not load the digit recognition model")
    st.error(
        "The recognition model could not be loaded. Make sure "
        "digit_model.keras is present and valid, then restart the app."
    )
    st.stop()

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

def preprocess_canvas(image_data):
    image_data = np.asarray(image_data)
    if image_data.ndim not in (2, 3):
        raise ValueError("Canvas image must be a grayscale or color image.")
    if image_data.ndim == 3 and image_data.shape[2] not in (3, 4):
        raise ValueError("Canvas image must have three or four color channels.")
    if not image_data.shape[0] or not image_data.shape[1]:
        raise ValueError("Canvas image is empty.")
    if image_data.shape[0] > 1024 or image_data.shape[1] > 1024:
        raise ValueError("Canvas image is larger than the supported size.")
    if image_data.dtype.kind not in "uif" or not np.isfinite(image_data).all():
        raise ValueError("Canvas image contains invalid pixel values.")

    image_data = np.clip(image_data, 0, 255).astype("uint8")
    img = Image.fromarray(image_data).convert("L")
    arr = np.array(img)

    coords = np.column_stack(np.where(arr > 20))
    if coords.size == 0:
        return None

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    digit = arr[y0:y1 + 1, x0:x1 + 1]

    # scale so the longer side fits in a 20x20 box, like MNIST's own preprocessing
    h, w = digit.shape
    scale = 20.0 / max(h, w)
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
    digit_img = Image.fromarray(digit).resize((new_w, new_h))

    # paste centered into a 28x28 black canvas
    canvas28 = Image.new("L", (28, 28), 0)
    offset = ((28 - new_w) // 2, (28 - new_h) // 2)
    canvas28.paste(digit_img, offset)

    return np.array(canvas28)


if st.button("Predict"):
    try:
        if canvas_result.image_data is None:
            st.warning("The drawing canvas is unavailable. Refresh the page and try again.")
            st.stop()

        processed = preprocess_canvas(canvas_result.image_data)
        if processed is None:
            st.warning("Draw a digit first.")
            st.stop()

        img_arr = (processed.astype("float32") / 255.0).reshape(1, 28, 28, 1)
        predictions = np.asarray(model.predict(img_arr, verbose=0), dtype="float32")

        if predictions.shape != (1, NUM_CLASSES):
            raise ValueError("Model returned an unexpected prediction shape.")
        if not np.isfinite(predictions).all():
            raise ValueError("Model returned non-finite predictions.")
        if np.any(predictions < 0) or np.any(predictions > 1):
            raise ValueError("Model returned values outside the probability range.")
        if not np.isclose(predictions.sum(), 1.0, atol=1e-3):
            raise ValueError("Model returned probabilities that do not sum to one.")

    except Exception:
        LOGGER.exception("Could not process the drawing or run prediction")
        st.error("The drawing could not be processed. Please try again with a new drawing.")
        st.stop()

    st.image(processed, caption="What the model sees (28x28)", width=100)
    digit = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]))

    st.subheader(f"Predicted digit: {digit}")
    st.write(f"Confidence: {confidence:.2%}")
    st.bar_chart(predictions[0])
