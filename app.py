from flask import Flask, request, jsonify, render_template, send_from_directory
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///crops.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Load your TensorFlow model (adjust path as needed)
# model = tf.keras.models.load_model('path/to/your/model')

# Database Model
class CropPrediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_name = db.Column(db.String(100))
    prediction = db.Column(db.String(50))
    confidence = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, server_default=db.func.now())
    
    def to_dict(self):
        return {
            'id': self.id,
            'image_name': self.image_name,
            'prediction': self.prediction,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }

# Root Route
@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <title>Crop Analysis System</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <meta name="description" content="AI-powered crop disease detection and analysis platform">
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
                animation: fadeIn 1s ease-in;
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
            }
            .btn:hover { 
                background: #45a049; 
                transform: scale(1.05);
                box-shadow: 0 4px 8px rgba(0,0,0,0.3);
            }
            .status {
                background: #e8f5e9;
                border-left: 4px solid #4CAF50;
                padding: 10px;
                margin: 20px 0;
                border-radius: 4px;
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(-20px); }
                to { opacity: 1; transform: translateY(0); }
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🌾 Crop Analysis System</h1>
            <p>AI-Powered Crop Disease Detection and Analysis Platform</p>
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
        
        <div style="text-align: center; margin-top: 30px;">
            <button class="btn" onclick="window.location.href='/predict'">
                🚀 Try Prediction Now
            </button>
        </div>
        
        <div style="text-align: center; margin-top: 20px; color: #666;">
            <p>Version 1.0.0 | Last Updated: """ + datetime.now().strftime("%Y-%m-%d") + """</p>
        </div>
    </body>
    </html>
    """

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

# Prediction Endpoint
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return ""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <title>Crop Prediction - Upload Image</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <link rel="icon" type="image/x-icon" href="/favicon.ico">
            <style>
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
                    margin-top: 50px;
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
                }
                .back-link {
                    color: #4CAF50;
                    text-decoration: none;
                    font-weight: bold;
                }
                .back-link:hover {
                    text-decoration: underline;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🌱 Crop Disease Detection</h1>
                <p>Upload an image of your crop for AI-powered analysis</p>
                
                <div class="upload-area">
                    <p>📸 Select an image file (JPG, PNG, etc.)</p>
                    <form method="post" enctype="multipart/form-data">
                        <input type="file" name="image" accept="image/*" required><br><br>
                        <button type="submit" class="btn">🔍 Analyze Crop</button>
                    </form>
                </div>
                
                <p style="text-align: center; margin-top: 30px;">
                    <a href="/" class="back-link">← Back to Home</a>
                </p>
            </div>
        </body>
        </html>
        "
    
    # Handle POST request for image upload
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
        
        # Log the prediction request
        logger.info(f"Processing image: {file.filename}, Size: {image.size}")
        
        # Preprocess image for your model
        # image = preprocess_image(image)
        # prediction = model.predict(image)
        
        # For now, return a mock response
        result = {
            "prediction": "Healthy",
            "confidence": 0.95,
            "recommendations": [
                "Continue current care routine",
                "Monitor for any changes",
                "Apply fertilizer as scheduled",
                "Maintain proper watering schedule"
            ],
            "image_info": {
                "filename": file.filename,
                "size": f"{image.size[0]}x{image.size[1]}",
                "format": image.format
            }
        }
        
        # Save to database
        try:
            db_record = CropPrediction(
                image_name=file.filename,
                prediction=result['prediction'],
                confidence=result['confidence']
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

# History Endpoint
@app.route('/history')
def history():
    try:
        # Get recent predictions from database
        predictions = CropPrediction.query.order_by(CropPrediction.timestamp.desc()).limit(20).all()
        return jsonify({
            "predictions": [p.to_dict() for p in predictions],
            "total": len(predictions)
        })
    except Exception as e:
        logger.error(f"Error fetching history: {str(e)}")
        # Return mock data if database fails
        mock_predictions = [
            {"id": 1, "date": "2025-01-01", "prediction": "Healthy", "confidence": 0.92},
            {"id": 2, "date": "2025-01-02", "prediction": "Early Blight", "confidence": 0.87}
        ]
        return jsonify({
            "predictions": mock_predictions,
            "total": len(mock_predictions),
            "note": "Using mock data - database connection issue"
        })

# Catch-all route for undefined paths
@app.route('/<path:path>')
def catch_all(path):
    logger.warning(f"404 - Path not found: {path}")
    return jsonify({
        "error": f"Path '{path}' not found",
        "available_endpoints": [
            "/",
            "/predict",
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
    return jsonify({"error": "Bad request"}), 400

# Create database tables
with app.app_context():
    try:
        db.create_all()
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

