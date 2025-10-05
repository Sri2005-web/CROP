import os
import uuid
from flask import Flask, request, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image, ImageStat
import numpy as np
import random

# --- CONFIGURATION ---
app = Flask(__name__, static_folder=None)
CORS(app)

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# File Upload Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- DATABASE MODELS ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_path = db.Column(db.String(120), nullable=False)
    disease_name = db.Column(db.String(120), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    treatment = db.Column(db.Text, nullable=False)

# --- DISEASE LIBRARY & TREATMENTS ---
# This library defines the "symptoms" our simulated AI will look for.
DISEASE_LIBRARY = {
    "Tomato Early Blight": {
        "symptoms": {"brown_spots": 0.4, "yellowing": 0.2, "green": 0.4},
        "treatment": "Apply a copper-based fungicide. Remove affected leaves and ensure good air circulation. Avoid overhead watering."
    },
    "Powdery Mildew": {
        "symptoms": {"white_dust": 0.6, "green": 0.4},
        "treatment": "Treat with neem oil or a sulfur-based fungicide. Increase air circulation and reduce humidity around plants."
    },
    "Leaf Rust": {
        "symptoms": {"rusty_spots": 0.5, "yellowing": 0.3, "green": 0.2},
        "treatment": "Remove affected leaves. Apply a fungicide suitable for rust, and ensure proper spacing between plants."
    },
    "Nitrogen Deficiency": {
        "symptoms": {"yellowing": 0.7, "green": 0.3},
        "treatment": "Apply a nitrogen-rich fertilizer. Ensure soil is not waterlogged, which can prevent nutrient uptake."
    },
    "Healthy": {
        "symptoms": {"green": 0.9},
        "treatment": "No treatment needed. Keep up the good care!"
    }
}

# --- CORE PREDICTION LOGIC (ADVANCED SIMULATION) ---
def analyze_image_for_leaf_and_disease(image_path):
    """
    Performs a simulated, but intelligent, analysis of the image.
    1. Checks if it's a leaf.
    2. Analyzes for visual "symptoms".
    3. Matches symptoms against the disease library for the best fit.
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)

        # --- 1. Leaf Detection ---
        pixels = img_array.reshape(-1, 3)
        green_pixels = pixels[(pixels[:, 1] > pixels[:, 0] * 1.1) & (pixels[:, 1] > pixels[:, 2] * 1.1)]
        green_ratio = len(green_pixels) / len(pixels)
        if green_ratio < 0.20: # Increased threshold slightly for better accuracy
            return {"is_leaf": False, "error": "Invalid image: Please upload a clear image of a crop leaf."}

        # --- 2. Symptom Analysis Simulation ---
        # Analyze the image to find the percentage of different "symptom" colors
        total_pixels = len(pixels)

        # Define color ranges for symptoms (these are heuristics)
        # Brown/Spots
        brown_mask = (pixels[:, 0] > 90) & (pixels[:, 1] > 60) & (pixels[:, 2] < 90)
        brown_ratio = np.sum(brown_mask) / total_pixels

        # Yellowing
        yellow_mask = (pixels[:, 0] > 150) & (pixels[:, 1] > 150) & (pixels[:, 2] < 100)
        yellow_ratio = np.sum(yellow_mask) / total_pixels

        # White/Dust (for Powdery Mildew)
        white_mask = (pixels[:, 0] > 200) & (pixels[:, 1] > 200) & (pixels[:, 2] > 200)
        white_ratio = np.sum(white_mask) / total_pixels

        # Rusty Spots
        rusty_mask = (pixels[:, 0] > 150) & (pixels[:, 1] > 80) & (pixels[:, 2] < 80)
        rusty_ratio = np.sum(rusty_mask) / total_pixels

        # Green
        green_mask = (pixels[:, 1] > pixels[:, 0] * 1.1) & (pixels[:, 1] > pixels[:, 2] * 1.1)
        green_ratio_symptom = np.sum(green_mask) / total_pixels

        # Create a "symptom profile" for the current image
        image_symptoms = {
            "green": green_ratio_symptom,
            "brown_spots": brown_ratio,
            "yellowing": yellow_ratio,
            "white_dust": white_ratio,
            "rusty_spots": rusty_ratio
        }

        # --- 3. Matching with Disease Library ---
        best_match = None
        lowest_error = float('inf')

        for disease, data in DISEASE_LIBRARY.items():
            # Calculate a "distance" or "error" between the image symptoms and the disease symptoms
            error = 0
            for symptom, value in data["symptoms"].items():
                error += abs(image_symptoms.get(symptom, 0) - value)

            if error < lowest_error:
                lowest_error = error
                best_match = disease

        # Calculate confidence based on how good the match is (lower error = higher confidence)
        # This is a simple way to simulate confidence scores
        confidence = max(75, 98 - (lowest_error * 100))
        confidence = round(confidence + random.uniform(-2, 2), 2) # Add small randomness

        return {
            "is_leaf": True,
            "disease": best_match,
            "confidence": confidence,
            "treatment": DISEASE_LIBRARY[best_match]["treatment"]
        }

    except Exception as e:
        return {"is_leaf": False, "error": f"Could not process image: {e}"}


# --- API ENDPOINTS (No changes needed here) ---

@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    if User.query.filter_by(username=data['username']).first():
        return jsonify({'error': 'Username already exists'}), 400
    new_user = User(username=data['username'])
    db.session.add(new_user)
    db.session.commit()
    return jsonify({'message': 'User registered successfully', 'user_id': new_user.id}), 201

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    user = User.query.filter_by(username=data['username']).first()
    if user:
        return jsonify({'message': 'Login successful', 'user_id': user.id}), 200
    return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'image' not in request.files or 'user_id' not in request.form:
        return jsonify({'error': 'Image file and user_id are required'}), 400

    file = request.files['image']
    user_id = request.form['user_id']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        filename = str(uuid.uuid4()) + ".jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        analysis = analyze_image_for_leaf_and_disease(filepath)

        if not analysis["is_leaf"]:
            os.remove(filepath)
            return jsonify({'error': analysis['error']}), 400

        new_prediction = Prediction(
            user_id=user_id,
            image_path=filepath,
            disease_name=analysis["disease"],
            confidence=analysis["confidence"],
            treatment=analysis["treatment"]
        )
        db.session.add(new_prediction)
        db.session.commit()

        return jsonify({
            'disease': analysis["disease"],
            'confidence': f"{analysis['confidence']}%",
            'treatment': analysis["treatment"]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'error': 'user_id is required'}), 400
    predictions = Prediction.query.filter_by(user_id=user_id).order_by(Prediction.id.desc()).all()
    history = []
    for p in predictions:
        history.append({'timestamp': p.id, 'disease': p.disease_name, 'confidence': f"{p.confidence}%"})
    return jsonify(history)

# --- SERVE THE FRONTEND ---
@app.route('/')
def serve_index():
    frontend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'frontend')
    return send_from_directory(frontend_path, 'index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)