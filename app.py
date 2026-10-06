import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow import keras
from PIL import Image

st.title("Handwritten Digit Recognition")


@st.cache_resource
def load_model():
    return keras.models.load_model("digit_model.keras")


model = load_model()

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
    return_image_data=True,
)

def preprocess_canvas(image_data):
    img = Image.fromarray(image_data.astype("uint8")).convert("L")
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
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
    digit_img = Image.fromarray(digit).resize((new_w, new_h))

    # paste centered into a 28x28 black canvas
    canvas28 = Image.new("L", (28, 28), 0)
    offset = ((28 - new_w) // 2, (28 - new_h) // 2)
    canvas28.paste(digit_img, offset)

    return np.array(canvas28)


if canvas_result.image_data is not None and st.button("Predict"):
    processed = preprocess_canvas(canvas_result.image_data)

    if processed is None:
        st.warning("Draw a digit first.")
        st.stop()

    st.image(processed, caption="What the model sees (28x28)", width=100)

    img_arr = processed.astype("float32") / 255.0
    img_arr = img_arr.reshape(1, 28, 28, 1)

    pred = model.predict(img_arr)
    digit = int(np.argmax(pred))
    confidence = float(np.max(pred))

    st.subheader(f"Predicted digit: {digit}")
    st.write(f"Confidence: {confidence:.2%}")
    st.bar_chart(pred[0])