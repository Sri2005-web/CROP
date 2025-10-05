from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for, session
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import os
import logging
from datetime import datetime
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Secret key for session management
app.secret_key = os.environ.get('SECRET_KEY', 'crop_analysis_secret_key_12345')

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///crops.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Load your TensorFlow model (adjust path as needed)
# model = tf.keras.models.load_model('path/to/your/model')

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, server_default=db.func.now())

class CropPrediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_name = db.Column(db.String(100))
    prediction = db.Column(db.String(50))
    confidence = db.Column(db.Float)
    treatment = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, server_default=db.func.now())
    
    def to_dict(self):
        return {
            'id': self.id,
            'image_name': self.image_name,
            'prediction': self.prediction,
            'confidence': self.confidence,
            'treatment': self.treatment,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }

# Helper function to check if image is a plant/crop/leaf
def is_plant_image(image):
    """Check if the uploaded image is likely a plant, crop, or leaf"""
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Get image dimensions
        width, height = image.size
        
        # Simple heuristic: plant images usually have more green pixels
        # This is a basic check - in a real app, you'd use a more sophisticated method
        pixels = np.array(image)
        
        # Calculate the percentage of green pixels
        green_pixels = np.sum((pixels[:,:,1] > pixels[:,:,0]) & (pixels[:,:,1] > pixels[:,:,2]))
        total_pixels = width * height
        green_percentage = (green_pixels / total_pixels) * 100
        
        # If more than 15% of pixels are predominantly green, consider it a plant image
        return green_percentage > 15
        
    except Exception as e:
        logger.error(f"Error checking plant image: {str(e)}")
        return False

# Login Page
@app.route('/')
def index():
    # If user is already logged in, redirect to dashboard
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <title>Crop Analysis System - Login</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="icon" type="image/x-icon" href="/favicon.ico">
    <style>
        * { box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            max-width: 500px; 
            margin: 0 auto; 
            padding: 20px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .login-container {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .header { 
            text-align: center; 
            color: #2E7D32; 
            margin-bottom: 30px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: bold;
            color: #333;
        }
        input[type="text"], input[type="password"] {
            width: 100%;
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
        }
        .btn { 
            background: #4CAF50; 
            color: white; 
            padding: 12px 20px; 
            border: none; 
            border-radius: 25px; 
            cursor: pointer; 
            font-size: 16px;
            transition: all 0.3s ease;
            width: 100%;
        }
        .btn:hover { 
            background: #45a049; 
        }
        .error {
            color: #D32F2F;
            margin-top: 10px;
            text-align: center;
        }
        .info {
            background: #e8f5e9;
            border-left: 4px solid #4CAF50;
            padding: 10px;
            margin-bottom: 20px;
            border-radius: 4px;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="login-container">
        <div class="header">
            <h1>🌾 Crop Analysis System</h1>
            <p>Login to access the system</p>
        </div>
        
        <div class="info">
            <strong>Demo Credentials:</strong><br>
            Username: 123456<br>
            Password: 1234
        </div>
        
        <form method="post" action="/login">
            <div class="form-group">
                <label for="username">Username:</label>
                <input type="text" id="username" name="username" required>
            </div>
            
            <div class="form-group">
                <label for="password">Password:</label>
                <input type="password" id="password" name="password" required>
            </div>
            
            <button type="submit" class="btn">Login</button>
        </form>
        
        <div id="error-message" class="error"></div>
    </div>
    
    <script>
        // Check for error parameter in URL
        const urlParams = new URLSearchParams(window.location.search);
        if (urlParams.get('error') === '1') {
            document.getElementById('error-message').textContent = 'Invalid username or password';
        }
    </script>
</body>
</html>'''

# Login handler
@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')
    
    # Simple authentication (in a real app, use proper password hashing)
    if username == '123456' and password == '1234':
        session['username'] = username
        return redirect(url_for('dashboard'))
    else:
        return redirect(url_for('index') + '?error=1')

# Logout handler
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('index'))

# Dashboard Page
@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('index'))
    
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <title>Crop Analysis System - Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="icon" type="image/x-icon" href="/favicon.ico">
    <style>
        * { box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            max-width: 900px; 
            margin: 0 auto; 
            padding: 20px; 
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
        }
        .header { 
            text-align: center; 
            color: #2E7D32; 
            margin-bottom: 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 {
            margin: 0;
        }
        .logout-btn {
            background: #f44336;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 20px;
            cursor: pointer;
            font-size: 14px;
        }
        .feature { 
            margin: 20px 0; 
            padding: 20px; 
            background: white; 
            border-radius: 10px; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }
        .feature:hover { transform: translateY(-5px); }
        .btn { 
            background: #4CAF50; 
            color: white; 
            padding: 12px 24px; 
            border: none; 
            border-radius: 25px; 
            cursor: pointer; 
            font-size: 16px;
            transition: all 0.3s ease;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            text-decoration: none;
            display: inline-block;
            margin: 5px;
        }
        .btn:hover { 
            background: #45a049; 
            transform: scale(1.05);
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        }
        .btn-primary {
            background: #2196F3;
        }
        .btn-primary:hover {
            background: #0b7dda;
        }
        .status {
            background: #e8f5e9;
            border-left: 4px solid #4CAF50;
            padding: 10px;
            margin: 20px 0;
            border-radius: 4px;
        }
        .welcome {
            text-align: center;
            margin-bottom: 20px;
            font-size: 18px;
        }
        .action-buttons {
            text-align: center;
            margin-top: 30px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🌾 Crop Analysis System</h1>
        <button class="logout-btn" onclick="window.location.href='/logout'">Logout</button>
    </div>
    
    <div class="welcome">
        Welcome, <strong>''' + session['username'] + '''</strong>!
    </div>
    
    <div class="status">
        <strong>✅ System Status:</strong> All systems operational
    </div>
    
    <div class="feature">
        <h3>🔍 Available Features:</h3>
        <ul>
            <li>Crop Disease Detection using Deep Learning</li>
            <li>Real-time Health Analysis</li>
            <li>Image Upload & Processing</li>
            <li>Webcam Capture Support</li>
            <li>Prediction History & Analytics</li>
            <li>Mobile-Friendly Interface</li>
        </ul>
    </div>
    
    <div class="feature">
        <h3>📡 API Endpoints:</h3>
        <ul>
            <li><strong>POST /predict</strong> - Upload image for crop analysis</li>
            <li><strong>GET /history</strong> - View prediction history</li>
            <li><strong>GET /health</strong> - Check system status</li>
            <li><strong>GET /stats</strong> - View system statistics</li>
        </ul>
    </div>
    
    <div class="action-buttons">
        <a href="/capture" class="btn btn-primary">📸 Capture Image</a>
        <a href="/upload" class="btn">📁 Upload Image</a>
        <a href="/history" class="btn">📊 View History</a>
    </div>
</body>
</html>'''

# Image Capture Page
@app.route('/capture')
def capture():
    if 'username' not in session:
        return redirect(url_for('index'))
    
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <title>Crop Analysis System - Capture Image</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="icon" type="image/x-icon" href="/favicon.ico">
    <style>
        * { box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            max-width: 800px; 
            margin: 0 auto; 
            padding: 20px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-top: 20px;
        }
        .header { 
            text-align: center; 
            color: #2E7D32; 
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 {
            margin: 0;
        }
        .back-link {
            color: #4CAF50;
            text-decoration: none;
            font-weight: bold;
        }
        .video-container {
            display: flex;
            justify-content: center;
            margin: 20px 0;
        }
        video {
            border: 2px solid #4CAF50;
            border-radius: 10px;
            max-width: 100%;
        }
        .controls {
            text-align: center;
            margin: 20px 0;
        }
        .btn { 
            background: #4CAF50; 
            color: white; 
            padding: 12px 24px; 
            border: none; 
            border-radius: 25px; 
            cursor: pointer; 
            font-size: 16px;
            transition: all 0.3s ease;
            margin: 5px;
        }
        .btn:hover { 
            background: #45a049; 
            transform: scale(1.05);
        }
        .btn-secondary {
            background: #f44336;
        }
        .btn-secondary:hover {
            background: #d32f2f;
        }
        .preview {
            text-align: center;
            margin: 20px 0;
        }
        #captured-image {
            max-width: 100%;
            border: 2px solid #4CAF50;
            border-radius: 10px;
            display: none;
        }
        .message {
            text-align: center;
            margin: 10px 0;
            padding: 10px;
            border-radius: 5px;
        }
        .error {
            background: #ffebee;
            color: #c62828;
        }
        .success {
            background: #e8f5e9;
            color: #2e7d32;
        }
        .hidden {
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📸 Capture Image</h1>
            <a href="/dashboard" class="back-link">← Back to Dashboard</a>
        </div>
        
        <div class="video-container">
            <video id="video" width="640" height="480" autoplay></video>
        </div>
        
        <div class="controls">
            <button id="start-camera" class="btn">Start Camera</button>
            <button id="capture-image" class="btn hidden">Capture Image</button>
            <button id="analyze-image" class="btn hidden">Analyze Image</button>
            <button id="retake" class="btn btn-secondary hidden">Retake</button>
        </div>
        
        <div class="preview">
            <canvas id="canvas" width="640" height="480" class="hidden"></canvas>
            <img id="captured-image" alt="Captured image">
        </div>
        
        <div id="message" class="message hidden"></div>
    </div>
    
    <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const capturedImage = document.getElementById('captured-image');
        const startCameraBtn = document.getElementById('start-camera');
        const captureImageBtn = document.getElementById('capture-image');
        const analyzeImageBtn = document.getElementById('analyze-image');
        const retakeBtn = document.getElementById('retake');
        const messageDiv = document.getElementById('message');
        
        let stream = null;
        let imageData = null;
        
        // Start camera
        startCameraBtn.addEventListener('click', async () => {
            try {
                stream = await navigator.mediaDevices.getUserMedia({ video: true });
                video.srcObject = stream;
                
                startCameraBtn.classList.add('hidden');
                captureImageBtn.classList.remove('hidden');
                
                showMessage('Camera started. Position your plant/leaf in the frame and capture.', 'success');
            } catch (err) {
                showMessage('Error accessing camera: ' + err.message, 'error');
            }
        });
        
        // Capture image
        captureImageBtn.addEventListener('click', () => {
            canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
            imageData = canvas.toDataURL('image/jpeg');
            capturedImage.src = imageData;
            capturedImage.style.display = 'block';
            video.style.display = 'none';
            
            captureImageBtn.classList.add('hidden');
            analyzeImageBtn.classList.remove('hidden');
            retakeBtn.classList.remove('hidden');
            
            showMessage('Image captured. Click "Analyze Image" to process.', 'success');
        });
        
        // Retake image
        retakeBtn.addEventListener('click', () => {
            capturedImage.style.display = 'none';
            video.style.display = 'block';
            
            captureImageBtn.classList.remove('hidden');
            analyzeImageBtn.classList.add('hidden');
            retakeBtn.classList.add('hidden');
            
            imageData = null;
            hideMessage();
        });
        
        // Analyze image
        analyzeImageBtn.addEventListener('click', () => {
            if (!imageData) return;
            
            // Convert data URL to blob
            const byteString = atob(imageData.split(',')[1]);
            const mimeString = imageData.split(',')[0].split(':')[1].split(';')[0];
            const ab = new ArrayBuffer(byteString.length);
            const ia = new Uint8Array(ab);
            
            for (let i = 0; i < byteString.length; i++) {
                ia[i] = byteString.charCodeAt(i);
            }
            
            const blob = new Blob([ab], { type: mimeString });
            
            // Create form data
            const formData = new FormData();
            formData.append('image', blob, 'captured-image.jpg');
            
            // Send to server
            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    showMessage('Error: ' + data.error, 'error');
                } else {
                    // Store result in session storage
                    sessionStorage.setItem('predictionResult', JSON.stringify(data));
                    window.location.href = '/result';
                }
            })
            .catch(error => {
                showMessage('Error: ' + error.message, 'error');
            });
        });
        
        function showMessage(text, type) {
            messageDiv.textContent = text;
            messageDiv.className = 'message ' + type;
            messageDiv.classList.remove('hidden');
        }
        
        function hideMessage() {
            messageDiv.classList.add('hidden');
        }
    </script>
</body>
</html>'''

# Image Upload Page
@app.route('/upload')
def upload():
    if 'username' not in session:
        return redirect(url_for('index'))
    
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <title>Crop Analysis System - Upload Image</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="icon" type="image/x-icon" href="/favicon.ico">
    <style>
        * { box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            max-width: 600px; 
            margin: 0 auto; 
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-top: 20px;
        }
        .header { 
            text-align: center; 
            color: #2E7D32; 
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 {
            margin: 0;
        }
        .back-link {
            color: #4CAF50;
            text-decoration: none;
            font-weight: bold;
        }
        .upload-area { 
            border: 3px dashed #4CAF50; 
            padding: 40px; 
            text-align: center; 
            margin: 20px 0; 
            border-radius: 10px;
            background: #f9f9f9;
            transition: all 0.3s ease;
        }
        .upload-area:hover {
            background: #e8f5e9;
            border-color: #45a049;
        }
        .btn { 
            background: #4CAF50; 
            color: white; 
            padding: 12px 30px; 
            border: none; 
            border-radius: 25px; 
            cursor: pointer; 
            font-size: 16px;
            transition: all 0.3s ease;
        }
        .btn:hover { 
            background: #45a049; 
            transform: scale(1.05);
        }
        input[type="file"] {
            margin: 10px 0;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            width: 100%;
        }
        .preview {
            text-align: center;
            margin: 20px 0;
        }
        #preview-image {
            max-width: 100%;
            border: 2px solid #4CAF50;
            border-radius: 10px;
            display: none;
        }
        .message {
            text-align: center;
            margin: 10px 0;
            padding: 10px;
            border-radius: 5px;
        }
        .error {
            background: #ffebee;
            color: #c62828;
        }
        .success {
            background: #e8f5e9;
            color: #2e7d32;
        }
        .hidden {
            display: none;
        }
        .note {
            background: #fff3e0;
            border-left: 4px solid #ff9800;
            padding: 10px;
            margin: 20px 0;
            border-radius: 4px;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📁 Upload Image</h1>
            <a href="/dashboard" class="back-link">← Back to Dashboard</a>
        </div>
        
        <div class="note">
            <strong>⚠️ Important:</strong> Please upload only plant, crop, or leaf images for accurate analysis.
        </div>
        
        <div class="upload-area">
            <p>📸 Select an image file (JPG, PNG, etc.)</p>
            <form id="upload-form" method="post" enctype="multipart/form-data">
                <input type="file" id="image-input" name="image" accept="image/*" required><br><br>
                <button type="submit" class="btn">🔍 Analyze Crop</button>
            </form>
        </div>
        
        <div class="preview">
            <img id="preview-image" alt="Preview image">
        </div>
        
        <div id="message" class="message hidden"></div>
    </div>
    
    <script>
        const imageInput = document.getElementById('image-input');
        const previewImage = document.getElementById('preview-image');
        const uploadForm = document.getElementById('upload-form');
        const messageDiv = document.getElementById('message');
        
        // Preview image when selected
        imageInput.addEventListener('change', function() {
            const file = this.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    previewImage.src = e.target.result;
                    previewImage.style.display = 'block';
                };
                reader.readAsDataURL(file);
            }
        });
        
        // Handle form submission
        uploadForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            
            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    showMessage('Error: ' + data.error, 'error');
                } else {
                    // Store result in session storage
                    sessionStorage.setItem('predictionResult', JSON.stringify(data));
                    window.location.href = '/result';
                }
            })
            .catch(error => {
                showMessage('Error: ' + error.message, 'error');
            });
        });
        
        function showMessage(text, type) {
            messageDiv.textContent = text;
            messageDiv.className = 'message ' + type;
            messageDiv.classList.remove('hidden');
        }
    </script>
</body>
</html>'''

# Prediction Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'username' not in session:
        return jsonify({"error": "Unauthorized access"}), 401
    
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No image selected"}), 400
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        return jsonify({"error": "File must be an image"}), 400
    
    # Process image
    try:
        # Read and process image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Check if image is a plant/crop/leaf
        if not is_plant_image(image):
            return jsonify({"error": "Please upload only plant, crop, or leaf images"}), 400
        
        # Log the prediction request
        logger.info(f"Processing image: {file.filename}, Size: {image.size}")
        
        # Preprocess image for your model
        # image = preprocess_image(image)
        # prediction = model.predict(image)
        
        # For now, return a mock response
        prediction_types = [
            {"name": "Healthy", "treatment": "Continue current care routine. Monitor for any changes. Apply fertilizer as scheduled. Maintain proper watering schedule."},
            {"name": "Early Blight", "treatment": "Apply copper-based fungicide. Remove affected leaves. Improve air circulation. Avoid overhead watering."},
            {"name": "Late Blight", "treatment": "Apply metalaxyl-based fungicide. Remove and destroy infected plants. Ensure proper drainage. Rotate crops annually."},
            {"name": "Powdery Mildew", "treatment": "Apply sulfur-based fungicide. Increase air circulation. Reduce humidity if possible. Remove affected parts."},
            {"name": "Leaf Spot", "treatment": "Apply chlorothalonil fungicide. Remove affected leaves. Avoid overhead watering. Ensure proper spacing between plants."}
        ]
        
        import random
        selected_prediction = random.choice(prediction_types)
        confidence = round(random.uniform(0.75, 0.98), 2)
        
        result = {
            "prediction": selected_prediction["name"],
            "confidence": confidence,
            "treatment": selected_prediction["treatment"],
            "image_info": {
                "filename": file.filename,
                "size": f"{image.size[0]}x{image.size[1]}",
                "format": image.format
            }
        }
        
        # Save to database
        try:
            user = User.query.filter_by(username=session['username']).first()
            db_record = CropPrediction(
                user_id=user.id,
                image_name=file.filename,
                prediction=result['prediction'],
                confidence=result['confidence'],
                treatment=result['treatment']
            )
            db.session.add(db_record)
            db.session.commit()
            logger.info(f"Saved prediction to database: {db_record.id}")
        except Exception as db_error:
            logger.error(f"Database error: {str(db_error)}")
            # Continue even if database save fails
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return jsonify({"error": f"Error processing image: {str(e)}"}), 500

# Result Page
@app.route('/result')
def result():
    if 'username' not in session:
        return redirect(url_for('index'))
    
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <title>Crop Analysis System - Result</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="icon" type="image/x-icon" href="/favicon.ico">
    <style>
        * { box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            max-width: 800px; 
            margin: 0 auto; 
            padding: 20px; 
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-top: 20px;
        }
        .header { 
            text-align: center; 
            color: #2E7D32; 
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 {
            margin: 0;
        }
        .back-link {
            color: #4CAF50;
            text-decoration: none;
            font-weight: bold;
        }
        .result-card {
            background: #f9f9f9;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .prediction {
            font-size: 24px;
            font-weight: bold;
            color: #2E7D32;
            text-align: center;
            margin: 20px 0;
        }
        .confidence {
            text-align: center;
            margin: 10px 0;
        }
        .confidence-bar {
            width: 100%;
            height: 20px;
            background: #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        .confidence-fill {
            height: 100%;
            background: #4CAF50;
        }
        .treatment {
            background: #e8f5e9;
            border-left: 4px solid #4CAF50;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }
        .treatment h3 {
            margin-top: 0;
            color: #2E7D32;
        }
        .btn { 
            background: #4CAF50; 
            color: white; 
            padding: 12px 24px; 
            border: none; 
            border-radius: 25px; 
            cursor: pointer; 
            font-size: 16px;
            transition: all 0.3s ease;
            text-decoration: none;
            display: inline-block;
            margin: 5px;
        }
        .btn:hover { 
            background: #45a049; 
            transform: scale(1.05);
        }
        .btn-secondary {
            background: #2196F3;
        }
        .btn-secondary:hover {
            background: #0b7dda;
        }
        .action-buttons {
            text-align: center;
            margin-top: 30px;
        }
        .loading {
            text-align: center;
            padding: 20px;
        }
        .error {
            text-align: center;
            color: #D32F2F;
            padding: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Analysis Result</h1>
            <a href="/dashboard" class="back-link">← Back to Dashboard</a>
        </div>
        
        <div id="loading" class="loading">
            <p>Loading results...</p>
        </div>
        
        <div id="error" class="error hidden">
            <p>Error loading results. Please try again.</p>
        </div>
        
        <div id="result-container" class="hidden">
            <div class="result-card">
                <div class="prediction" id="prediction"></div>
                
                <div class="confidence">
                    <strong>Confidence:</strong> <span id="confidence-value"></span>
                    <div class="confidence-bar">
                        <div class="confidence-fill" id="confidence-fill"></div>
                    </div>
                </div>
            </div>
            
            <div class="treatment">
                <h3>Recommended Treatment:</h3>
                <p id="treatment"></p>
            </div>
            
            <div class="action-buttons">
                <a href="/capture" class="btn btn-secondary">📸 New Analysis</a>
                <a href="/upload" class="btn btn-secondary">📁 Upload Another</a>
                <a href="/history" class="btn">📊 View History</a>
            </div>
        </div>
    </div>
    
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            // Get result from session storage
            const resultData = sessionStorage.getItem('predictionResult');
            
            if (resultData) {
                try {
                    const result = JSON.parse(resultData);
                    
                    // Hide loading
                    document.getElementById('loading').classList.add('hidden');
                    
                    // Show result
                    document.getElementById('result-container').classList.remove('hidden');
                    
                    // Fill in the data
                    document.getElementById('prediction').textContent = result.prediction;
                    document.getElementById('confidence-value').textContent = (result.confidence * 100).toFixed(1) + '%';
                    document.getElementById('confidence-fill').style.width = (result.confidence * 100) + '%';
                    document.getElementById('treatment').textContent = result.treatment;
                    
                    // Clear session storage
                    sessionStorage.removeItem('predictionResult');
                } catch (e) {
                    console.error('Error parsing result data:', e);
                    document.getElementById('loading').classList.add('hidden');
                    document.getElementById('error').classList.remove('hidden');
                }
            } else {
                // No result data found
                document.getElementById('loading').classList.add('hidden');
                document.getElementById('error').classList.remove('hidden');
            }
        });
    </script>
</body>
</html>'''

# History Page
@app.route('/history')
def history():
    if 'username' not in session:
        return redirect(url_for('index'))
    
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <title>Crop Analysis System - History</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="icon" type="image/x-icon" href="/favicon.ico">
    <style>
        * { box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            max-width: 900px; 
            margin: 0 auto; 
            padding: 20px; 
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-top: 20px;
        }
        .header { 
            text-align: center; 
            color: #2E7D32; 
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 {
            margin: 0;
        }
        .back-link {
            color: #4CAF50;
            text-decoration: none;
            font-weight: bold;
        }
        .history-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        .history-table th, .history-table td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        .history-table th {
            background-color: #4CAF50;
            color: white;
        }
        .history-table tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        .history-table tr:hover {
            background-color: #e8f5e9;
        }
        .confidence {
            font-weight: bold;
        }
        .high-confidence {
            color: #2E7D32;
        }
        .medium-confidence {
            color: #FF9800;
        }
        .low-confidence {
            color: #D32F2F;
        }
        .btn { 
            background: #4CAF50; 
            color: white; 
            padding: 8px 16px; 
            border: none; 
            border-radius: 20px; 
            cursor: pointer; 
            font-size: 14px;
            transition: all 0.3s ease;
            text-decoration: none;
            display: inline-block;
            margin: 5px;
        }
        .btn:hover { 
            background: #45a049; 
            transform: scale(1.05);
        }
        .action-buttons {
            text-align: center;
            margin-top: 30px;
        }
        .no-data {
            text-align: center;
            padding: 20px;
            color: #666;
        }
        .loading {
            text-align: center;
            padding: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Analysis History</h1>
            <a href="/dashboard" class="back-link">← Back to Dashboard</a>
        </div>
        
        <div id="loading" class="loading">
            <p>Loading history...</p>
        </div>
        
        <div id="history-container" class="hidden">
            <table class="history-table">
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Image Name</th>
                        <th>Prediction</th>
                        <th>Confidence</th>
                    </tr>
                </thead>
                <tbody id="history-tbody">
                </tbody>
            </table>
            
            <div id="no-data" class="no-data hidden">
                <p>No analysis history found. Start by capturing or uploading an image.</p>
            </div>
        </div>
        
        <div class="action-buttons">
            <a href="/capture" class="btn">📸 Capture Image</a>
            <a href="/upload" class="btn">📁 Upload Image</a>
        </div>
    </div>
    
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            fetch('/api/history')
            .then(response => response.json())
            .then(data => {
                document.getElementById('loading').classList.add('hidden');
                
                if (data.predictions && data.predictions.length > 0) {
                    document.getElementById('history-container').classList.remove('hidden');
                    
                    const tbody = document.getElementById('history-tbody');
                    tbody.innerHTML = '';
                    
                    data.predictions.forEach(prediction => {
                        const row = document.createElement('tr');
                        
                        // Format date
                        const date = new Date(prediction.timestamp);
                        const formattedDate = date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
                        
                        // Determine confidence class
                        let confidenceClass = 'high-confidence';
                        if (prediction.confidence < 0.8) confidenceClass = 'medium-confidence';
                        if (prediction.confidence < 0.6) confidenceClass = 'low-confidence';
                        
                        row.innerHTML = `
                            <td>${formattedDate}</td>
                            <td>${prediction.image_name}</td>
                            <td>${prediction.prediction}</td>
                            <td class="confidence ${confidenceClass}">${(prediction.confidence * 100).toFixed(1)}%</td>
                        `;
                        
                        tbody.appendChild(row);
                    });
                } else {
                    document.getElementById('no-data').classList.remove('hidden');
                }
            })
            .catch(error => {
                console.error('Error fetching history:', error);
                document.getElementById('loading').classList.add('hidden');
                document.getElementById('no-data').classList.remove('hidden');
            });
        });
    </script>
</body>
</html>'''

# API History Endpoint
@app.route('/api/history')
def api_history():
    if 'username' not in session:
        return jsonify({"error": "Unauthorized access"}), 401
    
    try:
        # Get user
        user = User.query.filter_by(username=session['username']).first()
        
        # Get recent predictions from database
        predictions = CropPrediction.query.filter_by(user_id=user.id).order_by(CropPrediction.timestamp.desc()).limit(20).all()
        return jsonify({
            "predictions": [p.to_dict() for p in predictions],
            "total": len(predictions)
        })
    except Exception as e:
        logger.error(f"Error fetching history: {str(e)}")
        return jsonify({"error": "Unable to fetch history"}), 500

# Favicon route to prevent 404 errors
@app.route('/favicon.ico')
def favicon():
    return "", 204  # No content response

# Robots.txt route
@app.route('/robots.txt')
def robots():
    return """User-agent: *
Allow: /
Disallow: /api/
Crawl-delay: 1
"""

# Health Check Endpoint
@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "Crop Analysis API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "database": "connected" if db.engine else "disconnected"
    })

# Statistics Endpoint
@app.route('/stats')
def stats():
    try:
        total_predictions = CropPrediction.query.count()
        recent_predictions = CropPrediction.query.order_by(CropPrediction.timestamp.desc()).limit(5).all()
        
        return jsonify({
            "total_predictions": total_predictions,
            "recent_predictions": [p.to_dict() for p in recent_predictions],
            "system_uptime": "24 hours"  # You can calculate actual uptime
        })
    except Exception as e:
        logger.error(f"Error fetching stats: {str(e)}")
        return jsonify({"error": "Unable to fetch statistics"}), 500

# Catch-all route for undefined paths
@app.route('/<path:path>')
def catch_all(path):
    logger.warning(f"404 - Path not found: {path}")
    return jsonify({
        "error": f"Path '{path}' not found",
        "available_endpoints": [
            "/",
            "/dashboard",
            "/capture",
            "/upload",
            "/result",
            "/history",
            "/health",
            "/stats"
        ]
    }), 404

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({"error": "Resource not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"500 error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(400)
def bad_request_error(error):
    return jsonify({"error": "Bad request"}), 500

# Create database tables
with app.app_context():
    try:
        db.create_all()
        
        # Create default user if not exists
        default_user = User.query.filter_by(username='123456').first()
        if not default_user:
            default_user = User(username='123456', password='1234')
            db.session.add(default_user)
            db.session.commit()
            logger.info("Created default user")
        
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {str(e)}")

# Run the app
if __name__ == '__main__':
    # Set debug to False for production
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    port = int(os.environ.get('PORT', 10000))
    
    logger.info(f"Starting Crop Analysis App on port {port}, debug={debug_mode}")
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
