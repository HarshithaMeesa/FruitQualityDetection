from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import numpy as np
import cv2
import base64
import os
from datetime import datetime
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Load your fine-tuned model
model = tf.keras.models.load_model("fruit_quality_multiclass_model.h5")
# Define the class mapping (order must match your training dataset)
class_indices = {
    0: "freshapple",
    1: "freshbanana",
    2: "freshorange",
    3: "rottenapple",
    4: "rottenbanana",
    5: "rottenorange"
}
class_names = list(class_indices.values())

# Image input size for the model
IMG_SIZE = 224

def prepare_image(image):
    # image comes in as a PIL image, convert it accordingly
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    # Expand dimensions to create batch of 1
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/")
def index():
    # Render the main page with the camera view and capture buttons
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get the captured image as a Base64 encoded string from the POST request
    encoded_data = request.form.get("imageData")
    if not encoded_data:
        return redirect(url_for("index"))
    
    # Remove header (data:image/jpeg;base64,...) if present
    header, encoded = encoded_data.split(",", 1)
    data = base64.b64decode(encoded)
    
    # Open the image with Pillow
    image = Image.open(BytesIO(data)).convert("RGB")
    
    # Prepare image for prediction
    processed_image = prepare_image(image)
    
    # Predict
    prediction = model.predict(processed_image)[0]
    pred_idx = np.argmax(prediction)
    predicted_class = class_names[pred_idx]
    confidence = np.max(prediction) * 100
    
    # Save image with timestamp, predicted result & confidence (if needed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"captured_{timestamp}_{predicted_class}_{int(confidence)}.jpg"
    save_dir = "Captured_images"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    image.save(save_path)
    
    # Render a results page with prediction and show the image (embedded as Base64)
    result_img_encoded = encoded_data  # we can re-use the captured image for the result page

    return render_template("result.html", 
                           predicted_class=predicted_class, 
                           confidence=confidence,
                           result_image=result_img_encoded)

if __name__ == '__main__':
    app.run(debug=True)
