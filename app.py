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

MODEL_PATH = 'letter.keras'  # Path to the model file

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

# Load model (on-the-fly when needed)
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found!")
    return keras.models.load_model(MODEL_PATH)

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

        # Load the model on-the-fly
        model = load_model()

        # Predict the letter
        prediction = model.predict(img)
        pred_letter_index = np.argmax(prediction, axis=-1)[0]

        pred_letter = ENCODER.inverse.get(pred_letter_index)

        if pred_letter is None:
            raise ValueError(f"Prediction {pred_letter_index} not found in encoder!")

        # Compare prediction to the expected letter
        correct = 'yes' if pred_letter == letter else 'no'
        next_letter = choice(list(ENCODER.keys()))

        return render_template("practice.html", letter=next_letter, correct=correct)

    except Exception as e:
        print(f"[ERROR] during practice: {e}")
        return render_template('error.html', error_message=str(e))

if __name__ == '__main__':
    app.run(debug=True)
