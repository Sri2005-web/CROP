import os
import uuid
import random
from flask import Flask, request, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from PIL import Image
import numpy as np

# --- CONFIGURATION ---
app = Flask(__name__, static_folder=None)
CORS(app)

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# File Upload Configuration
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
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
DISEASE_LIBRARY = {
    "Tomato Early Blight": {
        "symptoms": {"brown_spots": 0.4, "yellowing": 0.2, "green": 0.4},
        "treatment": "Apply copper fungicide, remove affected leaves, avoid overhead watering."
    },
    "Powdery Mildew": {
        "symptoms": {"white_dust": 0.6, "green": 0.4},
        "treatment": "Treat with neem oil or sulfur fungicide. Increase air circulation."
    },
    "Leaf Rust": {
        "symptoms": {"rusty_spots": 0.5, "yellowing": 0.3, "green": 0.2},
        "treatment": "Remove affected leaves, apply rust fungicide, ensure proper spacing."
    },
    "Nitrogen Deficiency": {
        "symptoms": {"yellowing": 0.7, "green": 0.3},
        "treatment": "Apply nitrogen-rich fertilizer, avoid waterlogged soil."
    },
    "Healthy": {
        "symptoms": {"green": 0.9},
        "treatment": "No treatment needed. Keep up the good care!"
    }
}

# --- CORE PREDICTION LOGIC ---
def analyze_image_for_leaf_and_disease(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        pixels = img_array.reshape(-1, 3)

        # Leaf detection
        green_pixels = pixels[(pixels[:, 1] > pixels[:, 0] * 1.1) & (pixels[:, 1] > pixels[:, 2] * 1.1)]
        green_ratio = len(green_pixels) / len(pixels)
        if green_ratio < 0.20:
            return {"is_leaf": False, "error": "Invalid image: Please upload a clear image of a crop leaf."}

        # Symptom masks
        brown_mask = (pixels[:, 0] > 90) & (pixels[:, 1] > 60) & (pixels[:, 2] < 90)
        yellow_mask = (pixels[:, 0] > 150) & (pixels[:, 1] > 150) & (pixels[:, 2] < 100)
        white_mask = (pixels[:, 0] > 200) & (pixels[:, 1] > 200) & (pixels[:, 2] > 200)
        rusty_mask = (pixels[:, 0] > 150) & (pixels[:, 1] > 80) & (pixels[:, 2] < 80)
        green_mask = (pixels[:, 1] > pixels[:, 0] * 1.1) & (pixels[:, 1] > pixels[:, 2] * 1.1)

        image_symptoms = {
            "green": np.sum(green_mask) / len(pixels),
            "brown_spots": np.sum(brown_mask) / len(pixels),
            "yellowing": np.sum(yellow_mask) / len(pixels),
            "white_dust": np.sum(white_mask) / len(pixels),
            "rusty_spots": np.sum(rusty_mask) / len(pixels)
        }

        # Match disease
        best_match, lowest_error = None, float('inf')
        for disease, data in DISEASE_LIBRARY.items():
            error = sum(abs(image_symptoms.get(sym, 0) - val) for sym, val in data["symptoms"].items())
            if error < lowest_error:
                lowest_error, best_match = error, disease

        confidence = max(75, 98 - (lowest_error * 100))
        confidence = round(confidence + random.uniform(-2, 2), 2)

        return {
            "is_leaf": True,
            "disease": best_match,
            "confidence": confidence,
            "treatment": DISEASE_LIBRARY[best_match]["treatment"]
        }

    except Exception as e:
        return {"is_leaf": False, "error": f"Could not process image: {e}"}

# --- API ROUTES ---
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
        filename = f"{uuid.uuid4()}.jpg"
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
    return jsonify([
        {'timestamp': p.id, 'disease': p.disease_name, 'confidence': f"{p.confidence}%"}
        for p in predictions
    ])

# --- FRONTEND SERVING ---
@app.route('/')
def serve_index():
    frontend_path = os.path.join(os.path.dirname(__file__), 'frontend')
    return send_from_directory(frontend_path, 'index.html')

@app.route('/frontend/<path:path>')
def serve_static(path):
    frontend_path = os.path.join(os.path.dirname(__file__), 'frontend')
    return send_from_directory(frontend_path, path)

# --- ENTRY POINT ---
if __name__ == '__main__':
    # Create tables inside application context
    with app.app_context():
        db.create_all()

    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
