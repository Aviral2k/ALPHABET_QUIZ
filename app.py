import os
import numpy as np
from bidict import bidict
from flask import (
    Flask, render_template, request,
    redirect, url_for, session
)
from random import choice
from tensorflow import keras

# Label encoder/decoder
ENCODER = bidict({
    'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6,
    'G': 7, 'H': 8, 'I': 9, 'J': 10, 'K': 11, 'L': 12,
    'M': 13, 'N': 14, 'O': 15, 'P': 16, 'Q': 17, 'R': 18,
    'S': 19, 'T': 20, 'U': 21, 'V': 22, 'W': 23, 'X': 24,
    'Y': 25, 'Z': 26
})

app = Flask(__name__)
app.secret_key = 'alphabet_quiz'

MODEL_PATH = 'letter.keras'  # saved model path

# Helper to initialize empty numpy files
def initialize_empty_files():
    os.makedirs('data', exist_ok=True)
    if not os.path.exists('data/labels.npy'):
        np.save('data/labels.npy', np.array([], dtype='<U1'))
    if not os.path.exists('data/images.npy'):
        np.save('data/images.npy', np.empty((0, 50, 50)))

# Initialize once
initialize_empty_files()

# Safe loader
def safe_load(file_path, default_value):
    try:
        return np.load(file_path, allow_pickle=True)
    except (EOFError, ValueError, FileNotFoundError):
        return default_value

# Train model if not exists
def train_model_if_needed():
    if not os.path.exists(MODEL_PATH):
        print("[INFO] Model not found. Training new model...")

        # Load existing data
        labels = safe_load('data/labels.npy', np.array([], dtype='<U1'))
        images = safe_load('data/images.npy', np.empty((0, 50, 50)))

        if len(labels) == 0 or len(images) == 0:
            raise RuntimeError("No training data available. Please add some samples first!")

        # Prepare data
        x_train = images.reshape(-1, 50, 50, 1)
        y_train = np.array([ENCODER[l] for l in labels])

        num_classes = len(ENCODER)

        # Build model
        model = keras.models.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 1)),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(num_classes + 1, activation='softmax')  # +1 because ENCODER values start from 1
        ])

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        # Train model
        model.fit(x_train, y_train, epochs=10)

        # Save model
        model.save(MODEL_PATH)
        print(f"[INFO] Model trained and saved to {MODEL_PATH}")
    else:
        print("[INFO] Found saved model, skipping training.")

# Training check before anything else
train_model_if_needed()

@app.route('/')
def index():
    session.clear()
    return render_template("index.html")

@app.route('/add-data', methods=['GET'])
def add_data_get():
    message = session.get('message', '')
    letter = choice(list(ENCODER.keys()))
    return render_template("addData.html", letter=letter, message=message)

@app.route('/add-data', methods=['POST'])
def add_data_post():
    try:
        label = request.form['letter']

        labels = safe_load('data/labels.npy', np.array([], dtype='<U1'))
        labels = np.append(labels, label)
        np.save('data/labels.npy', labels)

        pixels = request.form['pixels']
        pixels = pixels.split(',')
        img = np.array(pixels).astype(float).reshape(1, 50, 50)

        imgs = safe_load('data/images.npy', np.empty((0, 50, 50)))
        imgs = np.vstack([imgs, img])
        np.save('data/images.npy', imgs)

        session['message'] = f'"{label}" added to the training dataset'
        return redirect(url_for('add_data_get'))

    except Exception as e:
        print(f"[ERROR] while adding data: {e}")
        session['message'] = "Failed to add data."
        return redirect(url_for('add_data_get'))

@app.route('/practice', methods=['GET'])
def practice_get():
    letter = choice(list(ENCODER.keys()))
    return render_template("practice.html", letter=letter, correct='')

@app.route('/practice', methods=['POST'])
def practice_post():
    try:
        letter = request.form['letter']
        pixels = request.form['pixels']

        if not pixels:
            raise ValueError("No pixel data received!")

        pixels = pixels.split(',')
        img = np.array(pixels).astype(float).reshape(1, 50, 50, 1)

        model = keras.models.load_model(MODEL_PATH)
        prediction = model.predict(img)
        pred_letter_index = np.argmax(prediction, axis=-1)[0]

        pred_letter = ENCODER.inverse.get(pred_letter_index)

        if pred_letter is None:
            raise ValueError(f"Prediction {pred_letter_index} not found in encoder!")

        correct = 'yes' if pred_letter == letter else 'no'
        next_letter = choice(list(ENCODER.keys()))

        return render_template("practice.html", letter=next_letter, correct=correct)

    except Exception as e:
        print(f"[ERROR] during practice: {e}")
        return render_template('error.html', error_message=str(e))

if __name__ == '__main__':
    app.run(debug=True)
