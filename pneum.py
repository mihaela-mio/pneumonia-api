import os
import io
import uuid
import numpy as np
from datetime import datetime, timezone

from flask import Flask, request, jsonify
from PIL import Image

import tensorflow as tf
from google.cloud import datastore
from google.cloud import storage

# ----------------------------
# Initialize app
# ----------------------------
app = Flask(__name__)
# Class labels (adjust if your model differs)
CLASS_NAMES = ["Normal", "Pneumonia"]

# ----------------------------
# Load model
# ----------------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model_saved_model")
model = tf.saved_model.load(MODEL_DIR)
print(MODEL_DIR)
# Get default serving signature safely
infer = model.signatures["serving_default"]
                  

# ----------------------------
# Google Cloud clients
# ----------------------------
datastore_client = datastore.Client()
storage_client = storage.Client()

# ----------------------------
# Load image from GCS
# ----------------------------
def load_image_from_gcs(gcs_path: str):
    if not gcs_path.startswith("gs://"):
        raise ValueError("Invalid GCS path")

    path = gcs_path.replace("gs://", "")
    bucket_name, blob_name = path.split("/", 1)

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    image_bytes = blob.download_as_bytes()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))

    return image

# ----------------------------
# Preprocess image
# ----------------------------
def preprocess_image(image: Image.Image):
    image = np.array(image).astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ----------------------------
# Prediction endpoint
# ----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "gcs_path" not in data:
        return jsonify({"error": "Missing gcs_path"}), 400

    gcs_path = data["gcs_path"]

    try:
        # ----------------------------
        # Load + preprocess
        # ----------------------------
        image = load_image_from_gcs(gcs_path)
        image = preprocess_image(image)

        # ----------------------------
        # Inference
        # ----------------------------
        output = infer(tf.constant(image))

        # safer extraction
        output_tensor = next(iter(output.values()))
        preds = output_tensor.numpy()

        prob = float(preds.squeeze())

        # ----------------------------
        # Prediction logic
        # (assumes prob = Pneumonia probability)
        # ----------------------------
        if prob > 0.7:
            prediction = "Pneumonia"
            confidence = prob
        else:
            prediction = "Normal"
            confidence = 1 - prob

        # ----------------------------
        # Save to Datastore
        # ----------------------------
        entity_id = str(uuid.uuid4())
        key = datastore_client.key("Predictions", entity_id)

        entity = datastore.Entity(key=key)

        entity.update({
            "id": entity_id,
            "gcs_path": gcs_path,
            "image_name": os.path.basename(gcs_path),
            "prediction": prediction,
            "confidence": float(confidence),
            "probability_raw": prob,
            "timestamp": datetime.now(timezone.utc)
        })

        datastore_client.put(entity)

        print("Saved to Datastore project:", datastore_client.project)

        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        print("ERROR during prediction:", str(e))
        return jsonify({"error": str(e)}), 500

# ----------------------------
# Get predictions endpoint
# ----------------------------
@app.route("/predictions", methods=["GET"])
def get_predictions():
    try:
        query = datastore_client.query(kind="Predictions")
        results = list(query.fetch())

        return jsonify(results)

    except Exception as e:
        print("ERROR fetching predictions:", str(e))
        return jsonify({"error": str(e)}), 500

# ----------------------------
# Run locally
# ----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
