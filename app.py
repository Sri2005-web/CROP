import os
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import io
import base64
from datetime import datetime

# Initialize Flask App
app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///crop.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize Extensions
db = SQLAlchemy(app)
CORS(app)  # Enable CORS for all routes

# Database Models
class CropAnalysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_filename = db.Column(db.String(255), nullable=False)
    crop_type = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    health_status = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    notes = db.Column(db.Text)

    def to_dict(self):
        return {
            'id': self.id,
            'image_filename': self.image_filename,
            'crop_type': self.crop_type,
            'confidence': self.confidence,
            'health_status': self.health_status,
            'created_at': self.created_at.isoformat(),
            'notes': self.notes
        }

# Routes
@app.route('/')
def home():
    """Home route - fixes 404 error"""
    return jsonify({
        'message': 'CROP Analysis API is running!',
        'version': '1.0.0',
        'status': 'active',
        'endpoints': {
            'health': '/health',
            'analyze': '/analyze',
            'history': '/history',
            'docs': '/docs'
        }
    })

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'service': 'CROP API'
    })

@app.route('/docs')
def api_docs():
    """API documentation endpoint"""
    return jsonify({
        'title': 'CROP Analysis API',
        'version': '1.0.0',
        'description': 'API for crop disease detection and analysis',
        'endpoints': {
            'GET /': 'API information',
            'GET /health': 'Health check',
            'POST /analyze': 'Analyze crop image',
            'GET /history': 'Get analysis history',
            'GET /history/<id>': 'Get specific analysis',
            'DELETE /history/<id>': 'Delete analysis'
        }
    })

@app.route('/analyze', methods=['POST'])
def analyze_crop():
    """Analyze crop image for disease detection"""
    try:
        # Check if image is in request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Validate file type
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
        if not ('.' in file.filename and 
                file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, gif, bmp'}), 400
        
        # Secure filename and save
        filename = secure_filename(file.filename)
        
        # For demo purposes, simulate analysis
        # In production, you would integrate with your ML model here
        mock_results = {
            'crop_type': 'Tomato',
            'confidence': 0.92,
            'health_status': 'Healthy',
            'diseases': [],
            'recommendations': [
                'Continue regular watering',
                'Monitor for pests',
                'Apply fertilizer as needed'
            ]
        }
        
        # Save to database
        analysis = CropAnalysis(
            image_filename=filename,
            crop_type=mock_results['crop_type'],
            confidence=mock_results['confidence'],
            health_status=mock_results['health_status'],
            notes=f"Analysis completed on {datetime.utcnow().isoformat()}"
        )
        db.session.add(analysis)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'analysis_id': analysis.id,
            'results': mock_results,
            'message': 'Analysis completed successfully'
        })
        
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/history', methods=['GET'])
def get_analysis_history():
    """Get all crop analysis history"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        
        analyses = CropAnalysis.query.order_by(CropAnalysis.created_at.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        return jsonify({
            'analyses': [analysis.to_dict() for analysis in analyses.items],
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': analyses.total,
                'pages': analyses.pages,
                'has_next': analyses.has_next,
                'has_prev': analyses.has_prev
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to fetch history: {str(e)}'}), 500

@app.route('/history/<int:analysis_id>', methods=['GET'])
def get_analysis(analysis_id):
    """Get specific analysis by ID"""
    try:
        analysis = CropAnalysis.query.get_or_404(analysis_id)
        return jsonify(analysis.to_dict())
        
    except Exception as e:
        return jsonify({'error': f'Analysis not found: {str(e)}'}), 404

@app.route('/history/<int:analysis_id>', methods=['DELETE'])
def delete_analysis(analysis_id):
    """Delete specific analysis"""
    try:
        analysis = CropAnalysis.query.get_or_404(analysis_id)
        db.session.delete(analysis)
        db.session.commit()
        return jsonify({'message': 'Analysis deleted successfully'})
        
    except Exception as e:
        return jsonify({'error': f'Failed to delete analysis: {str(e)}'}), 500

# Error Handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(413)
def too_large(error):
    return jsonify({'error': 'File too large'}), 413

# Initialize Database
@app.before_first_request
def create_tables():
    db.create_all()

# Main execution
if __name__ == '__main__':
    # Get port from environment variable (Render sets this)
    port = int(os.environ.get('PORT', 5000))
    
    # Run the app
    app.run(host='0.0.0.0', port=port, debug=False)
