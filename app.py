import os
import numpy as np 
import tensorflow as tf 
import rasterio
import cv2 
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import io
import base64
import matplotlib.pyplot as plt 
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas 

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'tif', 'tiff'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to load and preprocess TIF images
def load_tif_image(image_file):
    with rasterio.open(image_file) as src:
        image = src.read()
        # Select three bands (nir, swir1 and Water Occurrence)
        image = image[[4, 5, 11], :, :]
        image = np.transpose(image, (1, 2, 0))
        image = cv2.resize(image, (128, 128))
        # Normalize
        image = image.astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min())
    return image

# Check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load the model
model = None

def load_model():
    global model
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'satellite_segmentation_model.h5')
    
    if not os.path.exists(model_path):
        print(f"Looking for model at: {model_path}")
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise
# Preprocess and predict
def predict_segmentation(image_path):
    # Load and preprocess the image
    img = load_tif_image(image_path)
    img_batch = np.expand_dims(img, axis=0)
    
    # Make prediction
    prediction = model.predict(img_batch)
    
    # Process prediction - Modified scaling
    prediction = prediction[0].squeeze()  # Remove batch dimension
    # Scale prediction to 0-1 range if needed
    prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min())
    # Apply threshold and scale to 0-255
    prediction_mask = (prediction > 0.5).astype(np.uint8) * 255
    
    # Generate visualization
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))  # Added third subplot to show raw prediction
    
    # Display the input image
    rgb_img = img.copy()
    rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
    axs[0].imshow(rgb_img)
    axs[0].set_title('Input Image')
    axs[0].axis('off')
    
    # Display the raw prediction (before thresholding)
    axs[1].imshow(prediction, cmap='jet')
    axs[1].set_title('Raw Prediction')
    axs[1].axis('off')
    
    # Display the thresholded prediction mask
    axs[2].imshow(prediction_mask, cmap='gray')
    axs[2].set_title('Thresholded Mask')
    axs[2].axis('off')
    
    plt.tight_layout()
    
    # Convert plot to image
    canvas = FigureCanvas(fig)
    img_buffer = io.BytesIO()
    canvas.print_png(img_buffer)
    plt.close(fig)
    
    # Encode the image to base64
    img_buffer.seek(0)
    img_str = base64.b64encode(img_buffer.read()).decode('utf-8')
    
    # Print some statistics for debugging
    print(f"Prediction stats - Min: {prediction.min():.4f}, Max: {prediction.max():.4f}, Mean: {prediction.mean():.4f}")
    print(f"Mask stats - Min: {prediction_mask.min()}, Max: {prediction_mask.max()}, Mean: {prediction_mask.mean():.2f}")
    
    return prediction_mask, img_str

# Routes
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Process the image and get prediction
            mask, img_b64 = predict_segmentation(filepath)
            
            # Return results
            return jsonify({
                'success': True,
                'visualization': img_b64,
                'message': 'Segmentation completed successfully!'
            })
            
        except Exception as e:
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
        
        finally:
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
    
    return jsonify({'error': 'File type not allowed. Please upload a .tif or .tiff file'}), 400

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('templates', exist_ok=True)
    
    # Load model before starting the server
    print("Loading model...")
    load_model()
    
    # Start Flask server
    print("Starting Flask server...")
    app.run(debug=True, port=5000)