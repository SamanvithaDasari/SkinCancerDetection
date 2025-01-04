from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import io
import numpy as np
import os

print("Current working directory:", os.getcwd())

app = Flask(__name__, template_folder="templates")

# Define the paths to your pre-trained models
model_paths = [
    'model1.h5', 
    'model_aug.keras', 
    'model_bal.keras', 
    'resnet_model_cnn.keras',
    'new_resnet_model.keras'
]

# Load all models
models = [load_model(path) for path in model_paths]

# Function to process image
def process_image(file_stream):
    img = image.load_img(io.BytesIO(file_stream.read()), target_size=(180, 180))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

@app.route('/')
def index():
    return render_template("indexx.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    img_array = process_image(file.stream)

    # Define your class labels
    class_labels = [
        'actinic keratosis', 'basal cell carcinoma', 'dermatofibroma', 'melanoma',
        'nevus', 'pigmented benign keratosis', 'seborrheic keratosis', 'squamous cell carcinoma', 'vascular lesion'
    ]

    # Store predictions from all models
    results = []

    for model in models:
        preds = model.predict(img_array)

        # Get the predicted class with the highest probability
        predicted_class_idx = np.argmax(preds[0])  # Index of the highest probability
        predicted_class = class_labels[predicted_class_idx]  # Get the corresponding class label

        # Append the predicted class for each model
        results.append({
            'model': model.name,  # Add model name to identify each model
            'predicted_class': predicted_class,
            'probability': float(preds[0][predicted_class_idx])  # Probability of the predicted class
        })

    return jsonify({'model_predictions': results})

if __name__ == '__main__':
    app.run(debug=True)
