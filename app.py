import os
import uuid
from flask import Flask, request, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

# --- CONFIGURATION ---
app = Flask(__name__, static_folder=None)
CORS(app)

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///database.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# File Upload Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- DATABASE MODELS ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    # Store a hashed password instead of a plain text one for security
    password_hash = db.Column(db.String(255), nullable=False)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_path = db.Column(db.String(120), nullable=False)
    disease_name = db.Column(db.String(120), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    treatment = db.Column(db.Text, nullable=False)

# --- TREATMENT INFORMATION ---
# This is kept on the backend to ensure the saved history has consistent treatment info.
TREATMENT_INFO = {
    "Tomato Early Blight": "Apply a copper-based fungicide. Remove affected leaves and ensure good air circulation.",
    "Apple Scab": "Apply fungicide in early spring. Rake and destroy fallen leaves to reduce spread.",
    "Healthy": "No treatment needed. Keep up the good care!",
    "Unknown": "Not found. Our system could not identify this disease."
}


# --- API ENDPOINTS ---

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    user = User.query.filter_by(username=data['username']).first()

    # Check if user exists AND if the provided password matches the stored hash
    if user and check_password_hash(user.password_hash, data['password']):
        return jsonify({'message': 'Login successful', 'user_id': user.id}), 200
    
    return jsonify({'error': 'Invalid username or password'}), 401

@app.route('/api/save-history', methods=['POST'])
def save_history():
    """
    Receives prediction results from the client-side AI model and saves them to the database.
    """
    if 'image' not in request.files or 'user_id' not in request.form:
        return jsonify({'error': 'Image file and user_id are required'}), 400

    file = request.files['image']
    user_id = request.form['user_id']
    disease_name = request.form.get('disease_name', 'Unknown')
    confidence = float(request.form.get('confidence', 0.0))

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # 1. Save the uploaded image
        filename = str(uuid.uuid4()) + ".jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # 2. Get the corresponding treatment information
        treatment = TREATMENT_INFO.get(disease_name, "No treatment info available.")

        # 3. Save the prediction record to the database
        new_prediction = Prediction(
            user_id=user_id,
            image_path=filepath,
            disease_name=disease_name,
            confidence=confidence,
            treatment=treatment
        )
        db.session.add(new_prediction)
        db.session.commit()

        # 4. Return a success response
        return jsonify({'message': 'History saved successfully.'}), 200

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
        history.append({
            'timestamp': p.id,
            'disease': p.disease_name,
            'confidence': f"{p.confidence}%"
        })
    return jsonify(history)


# --- SERVE THE FRONTEND ---
# This is the missing route that serves your main index.html page.
@app.route('/')
def serve_index():
    # This path correctly navigates from the 'backend' folder to the 'frontend' folder.
    frontend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'frontend')
    return send_from_directory(frontend_path, 'index.html')


if __name__ == '__main__':
    # Use a production server and bind to all addresses
    app.run(host='0.0.0.0', port=5000)
