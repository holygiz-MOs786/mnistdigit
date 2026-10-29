"""
Handwritten Digit Recognizer - training script
-----------------------------------------------
Trains a CNN on the MNIST dataset and saves everything the dashboard needs
into ONE file: model.pkl

Run on Google Colab (recommended, use a T4 GPU) or locally:
    python train.py
"""

import pickle

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix

EPOCHS = 10
BATCH_SIZE = 128
OUTPUT_FILE = "model.pkl"

keras.utils.set_random_seed(42)


# ---------- 1. Load data ----------
def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # Scale pixels 0-255 -> 0-1 and add a channel dimension: (28, 28) -> (28, 28, 1)
    x_train = (x_train.astype("float32") / 255.0)[..., np.newaxis]
    x_test = (x_test.astype("float32") / 255.0)[..., np.newaxis]
    return (x_train, y_train), (x_test, y_test)


# ---------- 2. Build the CNN ----------
def build_model():
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),

        # Block 1 - learns edges and simple strokes
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),

        # Block 2 - learns curves, loops and digit parts
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),

        # Classifier
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ], name="mnist_cnn")

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


# ---------- 3. Data augmentation ----------
augment_layers = keras.Sequential([
    layers.RandomRotation(0.05),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomZoom(0.1),
])


def augment(images, labels):
    return augment_layers(images, training=True), labels


def main():
    (x_train, y_train), (x_test, y_test) = load_data()
    print("Train:", x_train.shape, " Test:", x_test.shape)

    model = build_model()
    model.summary()

    train_ds = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
                .shuffle(10_000).batch(BATCH_SIZE)
                .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(tf.data.AUTOTUNE))

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]

    # ---------- 4. Train ----------
    history = model.fit(train_ds, epochs=EPOCHS,
                        validation_data=(x_test, y_test), callbacks=callbacks)

    # ---------- 5. Evaluate ----------
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    y_pred = model.predict(x_test, verbose=0).argmax(axis=1)
    print(f"\nTest accuracy: {test_acc:.4f}  |  Test loss: {test_loss:.4f}\n")
    print(classification_report(y_test, y_pred, digits=4))

    wrong = np.where(y_pred != y_test)[0][:24]  # keep some mistakes for the dashboard

    # ---------- 6. Save everything into model.pkl ----------
    bundle = {
        "model": model,
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True),
        "mistakes": {
            "images": (x_test[wrong].squeeze(-1) * 255).astype("uint8"),
            "true": y_test[wrong],
            "pred": y_pred[wrong],
        },
        "num_params": int(model.count_params()),
        "tf_version": tf.__version__,
    }
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Saved {OUTPUT_FILE}")

    # On Google Colab, download the file automatically
    try:
        from google.colab import files
        files.download(OUTPUT_FILE)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
