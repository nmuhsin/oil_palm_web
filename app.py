# ============================================================
# app.py  -  Oil Palm Ripeness Detection Web App
# Usage:  python app.py
# Then open:  http://127.0.0.1:5000
# ============================================================

import os
import uuid
import numpy as np
from flask import Flask, request, render_template, jsonify
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

# ── Config ───────────────────────────────────────────────────
UPLOAD_FOLDER    = os.path.join("static", "uploads")
ALLOWED_EXT      = {"jpg", "jpeg", "png", "JPG", "JPEG", "PNG"}
MAX_CONTENT_MB   = 16
CONFIDENCE_THRESHOLD = 85.0  # below this → reject as not oil palm

# ── Edit this path to point to your trained model ────────────
# FINAL DECISION (see Chapter 4, Section 4.7 robustness testing):
#   mobilenetv2_final.keras -> 93.24% accuracy, fastest (614s), and the
#   ONLY model with a perfect 4/4 pass rate rejecting out-of-distribution
#   images (grass, banana, hand, random object all correctly triggered
#   the <85% confidence warning). InceptionV3 (95.17% acc) failed 0/4 of
#   these tests, confidently misclassifying every OOD image at 95-99%
#   confidence. ResNet50V2 (90.82% acc) passed 3/4, failing once at 98.2%.
MODEL_PATH = r"C:\Users\nazif\Desktop\oil_palm_cnn\models\mobilenetv2_final.keras"
MODEL_INPUT_SIZE = (224, 224)   # auto-corrected to (299,299) below once the model loads; this is just the pre-load default

CLASS_NAMES  = ["Overripe", "Ripe", "Unripe"]
CLASS_MALAY  = ["Terlalu Masak", "Masak", "Belum Masak"]
CLASS_COLORS = ["#8a2e2e", "#c85a2e", "#6a8a3f"]
CLASS_ADVICE = [
    "Harvest immediately. Overripe fruit has elevated FFA content which reduces oil quality.",
    "Optimal harvest window. Fruit is at peak oil content and quality.",
    "Not ready for harvest. Allow more time for the fruit to ripen fully.",
]
CLASS_ICONS  = ["⏰", "✅", "🕐"]

app.config["UPLOAD_FOLDER"]      = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_MB * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Load model once at startup ────────────────────────────────
model = None

def load_model():
    global model, MODEL_INPUT_SIZE
    if not os.path.exists(MODEL_PATH):
        print(f"[WARNING] Model not found at: {MODEL_PATH}")
        print("          The app will run in DEMO mode with random predictions.")
        return

    print(f"[App] Loading model: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    # Auto-detect InceptionV3 by input shape
    if model.input_shape[1] == 299:
        MODEL_INPUT_SIZE = (299, 299)

    print(f"[App] Model loaded. Input size: {MODEL_INPUT_SIZE}")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1] in ALLOWED_EXT


def preprocess_image(image_path):
    """
    Load and resize image for model inference.
    ResNet50V2 / InceptionV3 / MobileNetV2 apply their own internal
    preprocessing. Do NOT normalize to [0,1] here.
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize(MODEL_INPUT_SIZE)
    arr = np.array(img, dtype=np.float32)   # keep raw [0,255]
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict(image_path):
    """
    Run inference. Returns dict with class, confidence, and all probabilities.
    Falls back to demo mode if no model is loaded.
    """
    if model is None:
        probs = np.random.dirichlet(np.ones(3)).tolist()
    else:
        arr   = preprocess_image(image_path)
        probs = model.predict(arr, verbose=0)[0].tolist()

    pred_idx   = int(np.argmax(probs))
    confidence = round(probs[pred_idx] * 100, 2)

    return {
        "class_idx":   pred_idx,
        "class_name":  CLASS_NAMES[pred_idx],
        "class_malay": CLASS_MALAY[pred_idx],
        "class_color": CLASS_COLORS[pred_idx],
        "class_icon":  CLASS_ICONS[pred_idx],
        "advice":      CLASS_ADVICE[pred_idx],
        "confidence":  confidence,
        "low_confidence": confidence < CONFIDENCE_THRESHOLD,
        "probabilities": [
            {
                "name":        CLASS_NAMES[i],
                "malay":       CLASS_MALAY[i],
                "probability": round(probs[i] * 100, 2),
                "color":       CLASS_COLORS[i],
            }
            for i in range(len(CLASS_NAMES))
        ],
        "demo_mode": model is None,
    }


# ── Routes ────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/detect")
def detect():
    return render_template("detect.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Upload a JPG or PNG image."}), 400

    ext      = file.filename.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    result = predict(save_path)
    result["image_url"] = f"/static/uploads/{filename}"

    return jsonify(result)


@app.route("/health")
def health():
    return jsonify({
        "status":       "ok",
        "model_loaded": model is not None,
        "model_path":   MODEL_PATH,
        "input_size":   list(MODEL_INPUT_SIZE),
        "threshold":    CONFIDENCE_THRESHOLD,
    })


# ── Entry point ───────────────────────────────────────────────

if __name__ == "__main__":
    load_model()
    print("\n[App] Starting server at http://127.0.0.1:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)