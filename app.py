from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import json
from PIL import Image
import io
import base64
import logging
from logging.handlers import RotatingFileHandler
from werkzeug.utils import secure_filename
from datetime import datetime
import time
from functools import wraps

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config.update(
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB max upload size
    ALLOWED_EXTENSIONS={'png', 'jpg', 'jpeg', 'gif'},
    MODEL_PATH="mobilenetv2_plant_model_final.keras",
    CLASS_NAMES_PATH='class_names.json',
    CACHE_TIMEOUT=3600  # 1 hour cache
)

# Set up logging
handler = RotatingFileHandler('plant_detection.log', maxBytes=10000, backupCount=3)
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)

# Global variables for model and class names
model = None
class_names = None
img_size = (256, 256)

# Metrics tracking
request_metrics = {
    'total_requests': 0,
    'successful_predictions': 0,
    'failed_predictions': 0,
    'avg_processing_time': 0
}

def load_model():
    """Load the ML model and class names with error handling"""
    global model, class_names
    
    try:
        start_time = time.time()
        app.logger.info("Loading model...")
        
        # Load model with custom objects if needed
        model = keras.models.load_model(
            app.config['MODEL_PATH'],
            compile=False
        )
        model.compile()  # Compile with default settings
        
        # Load class names
        with open(app.config['CLASS_NAMES_PATH'], 'r') as f:
            class_names = json.load(f)
            
        load_time = time.time() - start_time
        app.logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
        
    except Exception as e:
        app.logger.error(f"Error loading model: {str(e)}")
        raise e

# Load model at startup
load_model()

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def log_metrics(f):
    """Decorator to log request metrics"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        request_metrics['total_requests'] += 1
        
        try:
            response = f(*args, **kwargs)
            processing_time = time.time() - start_time
            
            # Update metrics
            request_metrics['successful_predictions'] += 1
            request_metrics['avg_processing_time'] = (
                request_metrics['avg_processing_time'] * (request_metrics['successful_predictions'] - 1) + processing_time
            ) / request_metrics['successful_predictions']
            
            app.logger.info(f"Processed request in {processing_time:.3f} seconds")
            return response
            
        except Exception as e:
            request_metrics['failed_predictions'] += 1
            app.logger.error(f"Request failed: {str(e)}")
            raise e
            
    return decorated_function

def preprocess_image(image):
    """Preprocess the image for model prediction"""
    try:
        # Resize and normalize image
        image = image.resize(img_size)
        img_array = np.array(image) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
        
    except Exception as e:
        app.logger.error(f"Error preprocessing image: {str(e)}")
        raise ValueError("Could not process the uploaded image")

def predict(image):
    """Make prediction on the processed image"""
    try:
        start_time = time.time()
        preprocessed = preprocess_image(image)
        
        # Make prediction
        prediction = model.predict(preprocessed)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction))
        
        app.logger.info(f"Prediction: {predicted_class} (Confidence: {confidence:.2%})")
        return predicted_class, confidence
        
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        raise ValueError("Error making prediction")

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict-image', methods=['POST'])
@log_metrics
def predict_image():
    """Handle image file upload and prediction"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            app.logger.warning("No file uploaded")
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        # Check if file has a name and is allowed
        if file.filename == '':
            app.logger.warning("No file selected")
            return jsonify({'error': 'No file selected'}), 400
            
        if not allowed_file(file.filename):
            app.logger.warning(f"Invalid file type: {file.filename}")
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Secure filename and log
        filename = secure_filename(file.filename)
        app.logger.info(f"Processing file: {filename}")
        
        # Open and convert image
        image = Image.open(file.stream).convert('RGB')
        
        # Make prediction
        predicted_class, confidence = predict(image)
        
        return jsonify({
            'class': predicted_class,
            'confidence': confidence,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        app.logger.error(f"Image prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict-webcam', methods=['POST'])
@log_metrics
def predict_webcam():
    """Handle webcam image data and prediction"""
    try:
        data_url = request.json.get('image')
        
        if not data_url:
            app.logger.warning("No image data received")
            return jsonify({'error': 'No image data received'}), 400
        
        # Extract base64 data
        try:
            header, encoded = data_url.split(",", 1)
            image_data = base64.b64decode(encoded)
        except Exception as e:
            app.logger.error("Invalid image data format")
            return jsonify({'error': 'Invalid image data format'}), 400
        
        # Open image
        try:
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
        except Exception as e:
            app.logger.error("Could not process image data")
            return jsonify({'error': 'Could not process image data'}), 400
        
        # Make prediction
        predicted_class, confidence = predict(image)
        
        return jsonify({
            'class': predicted_class,
            'confidence': confidence,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        app.logger.error(f"Webcam prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Endpoint to get service metrics"""
    return jsonify({
        'status': 'operational',
        'model_loaded': model is not None,
        'metrics': request_metrics,
        'last_updated': datetime.utcnow().isoformat()
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    app.logger.error(f"Server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Run with production settings
    app.run(host='0.0.0.0', port=5000, threaded=True)