"""
Handwritten Digit Recognizer - Streamlit dashboard
---------------------------------------------------
Loads the trained CNN from model.pkl and lets you:
  - draw a digit and get a live prediction
  - upload a photo of a handwritten digit
  - explore the model's performance (accuracy, training curves, confusion matrix)

Run:
    streamlit run app.py
"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

MODEL_FILE = "model.pkl"

st.set_page_config(page_title="Digit Recognizer", page_icon="✍️", layout="wide")


# ---------- Load model ----------
@st.cache_resource
def load_bundle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


if not os.path.exists(MODEL_FILE):
    st.error(f"`{MODEL_FILE}` not found. Run `python train.py` first and put the file next to app.py.")
    st.stop()

bundle = load_bundle(MODEL_FILE)
model = bundle["model"]


# ---------- Preprocessing ----------
def preprocess(img: Image.Image) -> np.ndarray:
    """Turn any image into MNIST format: 28x28, white digit on black, centred."""
    arr = np.array(img.convert("L"), dtype=np.float32)
    if arr.mean() > 127:  # dark ink on light paper -> invert
        arr = 255 - arr
    arr[arr < 50] = 0  # remove faint noise
    ys, xs = np.nonzero(arr)
    if len(xs) == 0:
        return np.zeros((1, 28, 28, 1), dtype=np.float32)
    arr = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]  # crop to the digit

    # Fit the digit into a 20x20 box (like MNIST), keeping its shape
    h, w = arr.shape
    scale = 20.0 / max(h, w)
    digit = Image.fromarray(arr.astype(np.uint8)).resize(
        (max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)

    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(digit, ((28 - digit.width) // 2, (28 - digit.height) // 2))
    out = np.array(canvas, dtype=np.float32)

    # Centre by centre of mass
    total = out.sum()
    if total > 0:
        cy = (out * np.arange(28)[:, None]).sum() / total
        cx = (out * np.arange(28)[None, :]).sum() / total
        out = np.roll(out, (int(round(14 - cy)), int(round(14 - cx))), axis=(0, 1))
    return (out / 255.0).reshape(1, 28, 28, 1)


def show_prediction(img: Image.Image):
    x = preprocess(img)
    if x.sum() == 0:
        st.info("Draw or upload a digit to see the prediction.")
        return
    probs = model.predict(x, verbose=0)[0]
    digit = int(probs.argmax())

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Prediction", digit, f"{probs[digit]:.1%} confident", delta_color="off")
        st.image(x.squeeze(), caption="What the model sees (28×28)", width=140, clamp=True)
    with c2:
        df = pd.DataFrame({"Digit": [str(i) for i in range(10)], "Probability": probs})
        st.bar_chart(df, x="Digit", y="Probability", height=260)
        if probs[digit] < 0.6:
            st.warning("Low confidence — try drawing bigger and centred.")


# ---------- Sidebar ----------
with st.sidebar:
    st.title("✍️ Digit Recognizer")
    st.caption("CNN trained on the MNIST dataset (70,000 handwritten digits).")
    st.metric("Test accuracy", f"{bundle['test_accuracy']:.2%}")
    st.metric("Test loss", f"{bundle['test_loss']:.4f}")
    st.metric("Parameters", f"{bundle['num_params']:,}")
    st.metric("Epochs trained", len(bundle["history"]["loss"]))
    st.divider()
    st.markdown("**Architecture**  \n"
                "Conv(32)×2 → Pool → Conv(64)×2 → Pool → Dense(128) → Softmax(10)")
    st.markdown("[GitHub repo](https://github.com/holygiz-MOs786)")

st.title("Handwritten Digit Recognizer")

tab_draw, tab_upload, tab_perf = st.tabs(["✏️ Draw", "📤 Upload image", "📊 Model performance"])

# ---------- Tab 1: Draw ----------
with tab_draw:
    left, right = st.columns([1, 2])
    with left:
        st.write("Draw a single digit (0–9):")
        canvas = st_canvas(
            stroke_width=18,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            return_image_data=True,
            key="canvas",
        )
        st.caption("Use the canvas toolbar to clear it.")
    with right:
        if canvas.image_data is not None:
            show_prediction(Image.fromarray(canvas.image_data.astype("uint8")).convert("RGB"))

# ---------- Tab 2: Upload ----------
with tab_upload:
    file = st.file_uploader("Upload a photo of a handwritten digit",
                            type=["png", "jpg", "jpeg", "bmp", "webp"])
    if file:
        img = Image.open(file)
        c1, c2 = st.columns([1, 3])
        c1.image(img, caption="Your image", width=180)
        with c2:
            show_prediction(img)
    else:
        st.caption("Tip: write a digit with a dark pen on white paper and take a close-up photo.")

# ---------- Tab 3: Performance ----------
with tab_perf:
    hist = pd.DataFrame(bundle["history"])
    hist.index = hist.index + 1
    hist.index.name = "Epoch"

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Accuracy per epoch")
        st.line_chart(hist[["accuracy", "val_accuracy"]].rename(
            columns={"accuracy": "Train", "val_accuracy": "Validation"}))
    with c2:
        st.subheader("Loss per epoch")
        st.line_chart(hist[["loss", "val_loss"]].rename(
            columns={"loss": "Train", "val_loss": "Validation"}))

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Confusion matrix")
        cm = bundle["confusion_matrix"]
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for i in range(10):
            for j in range(10):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.colorbar(im)
        fig.tight_layout()
        st.pyplot(fig)
    with c2:
        st.subheader("Per-digit scores")
        rep = bundle["report"]
        table = pd.DataFrame({
            "Digit": [str(d) for d in range(10)],
            "Precision": [rep[str(d)]["precision"] for d in range(10)],
            "Recall": [rep[str(d)]["recall"] for d in range(10)],
            "F1-score": [rep[str(d)]["f1-score"] for d in range(10)],
            "Test images": [int(rep[str(d)]["support"]) for d in range(10)],
        })
        st.dataframe(table.style.format({"Precision": "{:.2%}", "Recall": "{:.2%}",
                                         "F1-score": "{:.2%}"}),
                     hide_index=True, width="stretch")

    st.subheader("Some mistakes the model made")
    m = bundle["mistakes"]
    if len(m["images"]):
        cols = st.columns(8)
        for i, (im_, t, p) in enumerate(zip(m["images"], m["true"], m["pred"])):
            cols[i % 8].image(im_, caption=f"pred {p} · true {t}", width=70)
    else:
        st.write("No mistakes on the test set!")
