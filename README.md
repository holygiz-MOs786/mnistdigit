# ✍️ Handwritten Digit Recognizer — CNN + Streamlit Dashboard

A Convolutional Neural Network built with **TensorFlow/Keras** that recognizes handwritten digits (0–9), trained on the **MNIST** dataset (70,000 images, 28×28 grayscale). It comes with an interactive **Streamlit dashboard**: you can draw a digit, upload a photo, or explore how well the model performs.

## Features

- **CNN model**: two convolution blocks with BatchNorm, MaxPooling and Dropout (~468K parameters)
- **Data augmentation**: random rotation, shift and zoom, so the model handles real handwriting
- **Streamlit dashboard**:
  - ✏️ Draw a digit and get a live prediction with probabilities for every digit
  - 📤 Upload a photo of a handwritten digit
  - 📊 Model performance: accuracy and loss curves, confusion matrix, per-digit precision/recall, and the model's mistakes
- **Smart preprocessing**: any image is converted to MNIST format (inverted, cropped, resized, centred)

## Project structure

```
mnist-digit-recognizer/
├── train.py          # trains the CNN and saves model.pkl
├── app.py            # Streamlit dashboard
├── model.pkl         # trained model + metrics (created by train.py)
├── requirements.txt
└── README.md
```

## How to run

### 1. Train the model (Google Colab recommended)
1. Open [Google Colab](https://colab.research.google.com) and set **Runtime → Change runtime type → T4 GPU**.
2. Paste the contents of `train.py` into a cell and run it.
3. `model.pkl` downloads automatically. Put it in this folder.

Or train locally:
```bash
pip install -r requirements.txt
python train.py
```

### 2. Run the dashboard
```bash
pip install -r requirements.txt
streamlit run app.py
```
Open http://localhost:8501.

## Model architecture

```
Input (28×28×1)
→ Conv2D(32) → BatchNorm → Conv2D(32) → MaxPool → Dropout(0.25)
→ Conv2D(64) → BatchNorm → Conv2D(64) → MaxPool → Dropout(0.25)
→ Flatten → Dense(128) → Dropout(0.5) → Dense(10, softmax)
```
Optimizer: Adam · Loss: sparse categorical cross-entropy · Up to 10 epochs with early stopping

**Why a CNN?** A normal neural network looks at each pixel separately. Convolution filters slide over the image and learn patterns (edges, curves, loops), so the model recognises a digit wherever it appears in the image.

## Results

| Metric | Value |
|---|---|
| Test accuracy | **~99.4%** <!-- replace with your number --> |
| Test images | 10,000 |

<!-- Add a screenshot of your dashboard: ![Dashboard](screenshot.png) -->

## What I learned

- How images are represented as tensors and normalised for neural networks
- How convolution, pooling, batch normalisation and dropout work together
- Why data augmentation and preprocessing matter for real-world input
- Evaluating a classifier with a confusion matrix and per-class metrics
- Building and deploying an interactive ML dashboard with Streamlit

## Tech stack

Python · TensorFlow / Keras · NumPy · Pandas · Matplotlib · scikit-learn · Streamlit

## Author

**Ankush Verma** — [GitHub](https://github.com/holygiz-MOs786) · [LinkedIn](https://linkedin.com/in/ankush-verma-a2b560444)
