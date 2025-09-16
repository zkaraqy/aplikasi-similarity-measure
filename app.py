from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from werkzeug.utils import secure_filename
import json
from datetime import datetime
import math

app = Flask(__name__)
app.secret_key = 'similarity_measure_secret_key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['SAMPLE_FOLDER'] = 'static/sample_images'
app.config['PLOTS_FOLDER'] = 'static/plots'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PLOTS_FOLDER'], exist_ok=True)

# Global variables to store processing results
current_image = None
segmented_image = None
angle_signature = None
normalized_signature = None
reference_signatures = {}

def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_angle_signature_from_image(image_path):
    """Extract real angle signature from an image file"""
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to create binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply morphological operations to clean up
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            return None
        
        # Get the largest contour (main object)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate centroid
        M = cv2.moments(largest_contour)
        if M['m00'] == 0:
            return None
        
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        # Calculate angle signature - distance from centroid for each angle
        signature = []
        angles = np.arange(0, 360, 1)  # 360 points for 0-359 degrees
        
        for angle in angles:
            # Convert angle to radians
            rad = np.radians(angle)
            
            # Find distances for this angle
            distances = []
            for point in largest_contour:
                px, py = point[0]
                # Calculate angle from centroid to this point
                point_angle = np.degrees(np.arctan2(py - cy, px - cx))
                if point_angle < 0:
                    point_angle += 360
                
                # If this point is close to our target angle (within tolerance)
                angle_diff = abs(point_angle - angle)
                if angle_diff < 2 or angle_diff > 358:  # Handle wrap-around
                    distance = np.sqrt((px - cx)**2 + (py - cy)**2)
                    distances.append(distance)
            
            if distances:
                signature.append(max(distances))  # Take maximum distance for this angle
            else:
                # Interpolate from nearby angles if no direct match
                signature.append(0)
        
        # Smooth the signature to handle missing values
        signature = np.array(signature)
        for i in range(len(signature)):
            if signature[i] == 0:
                # Find nearest non-zero values for interpolation
                left_val = 0
                right_val = 0
                for j in range(1, 10):  # Look within 10 degrees
                    if i-j >= 0 and signature[i-j] > 0:
                        left_val = signature[i-j]
                        break
                for j in range(1, 10):
                    if i+j < len(signature) and signature[i+j] > 0:
                        right_val = signature[i+j]
                        break
                
                if left_val > 0 and right_val > 0:
                    signature[i] = (left_val + right_val) / 2
                elif left_val > 0:
                    signature[i] = left_val
                elif right_val > 0:
                    signature[i] = right_val
        
        return signature
        
    except Exception as e:
        print(f"Error extracting signature from {image_path}: {e}")
        return None

def get_sample_images():
    """Get list of sample images"""
    sample_path = app.config['SAMPLE_FOLDER']
    if not os.path.exists(sample_path):
        return []
    
    images = []
    for filename in os.listdir(sample_path):
        if allowed_file(filename):
            images.append(filename)
    return sorted(images)

@app.route('/')
def index():
    """Main application page"""
    sample_images = get_sample_images()
    return render_template('index.html', sample_images=sample_images)

@app.route('/load_image', methods=['POST'])
def load_image():
    """Load image from file upload or sample selection"""
    global current_image, segmented_image, angle_signature, normalized_signature
    
    try:
        # Reset processing results
        segmented_image = None
        angle_signature = None
        normalized_signature = None
        
        if 'file' in request.files and request.files['file'].filename:
            # Handle file upload
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Load image
                current_image = cv2.imread(filepath)
                if current_image is None:
                    return jsonify({'success': False, 'message': 'Failed to load image'})
                
                session['current_image_path'] = f"uploads/{filename}"
                
        elif 'sample_image' in request.form:
            # Handle sample image selection
            sample_name = request.form['sample_image']
            sample_path = os.path.join(app.config['SAMPLE_FOLDER'], sample_name)
            
            if os.path.exists(sample_path):
                current_image = cv2.imread(sample_path)
                if current_image is None:
                    return jsonify({'success': False, 'message': 'Failed to load sample image'})
                
                session['current_image_path'] = f"sample_images/{sample_name}"
            else:
                return jsonify({'success': False, 'message': 'Sample image not found'})
        
        else:
            return jsonify({'success': False, 'message': 'No image provided'})
        
        return jsonify({
            'success': True, 
            'message': 'Image loaded successfully',
            'image_path': session['current_image_path']
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error loading image: {str(e)}'})

@app.route('/segment_image', methods=['POST'])
def segment_image():
    """Perform image segmentation"""
    global current_image, segmented_image
    
    try:
        if current_image is None:
            return jsonify({'success': False, 'message': 'No image loaded'})
        
        # Convert to grayscale
        gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to create binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply morphological operations to clean up the image
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        segmented_image = binary
        
        # Save segmented image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        seg_filename = f"segmented_{timestamp}.png"
        seg_path = os.path.join(app.config['UPLOAD_FOLDER'], seg_filename)
        cv2.imwrite(seg_path, segmented_image)
        
        session['segmented_image_path'] = f"uploads/{seg_filename}"
        
        return jsonify({
            'success': True,
            'message': 'Image segmentation completed',
            'segmented_path': session['segmented_image_path']
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error in segmentation: {str(e)}'})

@app.route('/calculate_angle_signature', methods=['POST'])
def calculate_angle_signature():
    """Calculate angle signature for the segmented object"""
    global segmented_image, angle_signature
    
    try:
        if segmented_image is None:
            return jsonify({'success': False, 'message': 'No segmented image available'})
        
        # Find contours
        contours, _ = cv2.findContours(segmented_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            return jsonify({'success': False, 'message': 'No contours found in segmented image'})
        
        # Get the largest contour (assuming it's the main object)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate centroid
        M = cv2.moments(largest_contour)
        if M['m00'] == 0:
            return jsonify({'success': False, 'message': 'Cannot calculate centroid'})
        
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        centroid = (cx, cy)
        
        # Calculate angle signature
        signature = []
        angles = np.arange(0, 360, 1)  # 360 points for 0-359 degrees
        
        for angle in angles:
            # Convert angle to radians
            rad = np.radians(angle)
            
            # Find intersection with boundary
            distances = []
            for point in largest_contour:
                px, py = point[0]
                # Calculate angle from centroid to this point
                point_angle = np.degrees(np.arctan2(py - cy, px - cx))
                if point_angle < 0:
                    point_angle += 360
                
                # If this point is close to our target angle
                if abs(point_angle - angle) < 1:
                    distance = np.sqrt((px - cx)**2 + (py - cy)**2)
                    distances.append(distance)
            
            if distances:
                signature.append(max(distances))  # Take maximum distance for this angle
            else:
                signature.append(0)
        
        angle_signature = np.array(signature)
        
        # Create and save plot
        plt.figure(figsize=(10, 6))
        plt.plot(angles, angle_signature, 'b-', linewidth=2)
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Distance from Centroid')
        plt.title('Angle Distance Signature')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 360)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"angle_signature_{timestamp}.png"
        plot_path = os.path.join(app.config['PLOTS_FOLDER'], plot_filename)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        session['angle_signature_plot'] = f"plots/{plot_filename}"
        
        return jsonify({
            'success': True,
            'message': 'Angle signature calculated successfully',
            'plot_path': session['angle_signature_plot'],
            'signature_data': angle_signature.tolist()
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error calculating angle signature: {str(e)}'})

@app.route('/normalize_signature', methods=['POST'])
def normalize_signature():
    """Normalize the angle signature"""
    global angle_signature, normalized_signature
    
    try:
        if angle_signature is None:
            return jsonify({'success': False, 'message': 'No angle signature available'})
        
        # Normalize signature to range [0, 1]
        min_val = np.min(angle_signature)
        max_val = np.max(angle_signature)
        
        if max_val == min_val:
            normalized_signature = np.zeros_like(angle_signature)
        else:
            normalized_signature = (angle_signature - min_val) / (max_val - min_val)
        
        # Create and save normalized plot
        plt.figure(figsize=(10, 6))
        angles = np.arange(0, 360, 1)
        plt.plot(angles, normalized_signature, 'g-', linewidth=2)
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Normalized Distance')
        plt.title('Normalized Angle Distance Signature')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 360)
        plt.ylim(0, 1)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"normalized_signature_{timestamp}.png"
        plot_path = os.path.join(app.config['PLOTS_FOLDER'], plot_filename)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        session['normalized_signature_plot'] = f"plots/{plot_filename}"
        
        return jsonify({
            'success': True,
            'message': 'Signature normalized successfully',
            'plot_path': session['normalized_signature_plot'],
            'normalized_data': normalized_signature.tolist()
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error normalizing signature: {str(e)}'})

@app.route('/classify_similarity', methods=['POST'])
def classify_similarity():
    """Perform similarity classification using real angle signatures"""
    global normalized_signature, reference_signatures
    
    try:
        if normalized_signature is None:
            return jsonify({'success': False, 'message': 'No normalized signature available'})
        
        # Build reference signatures from actual sample images
        reference_signatures = {}  # Reset to ensure fresh extraction
        sample_images = get_sample_images()
        
        for sample_file in sample_images:
            # Extract real angle signature from each sample image
            sample_path = os.path.join(app.config['SAMPLE_FOLDER'], sample_file)
            ref_signature = extract_angle_signature_from_image(sample_path)
            
            if ref_signature is not None:
                # Normalize reference signature using same method as input
                if np.max(ref_signature) != np.min(ref_signature):
                    normalized_ref = (ref_signature - np.min(ref_signature)) / (np.max(ref_signature) - np.min(ref_signature))
                else:
                    normalized_ref = np.zeros_like(ref_signature)
                
                # Extract shape name from filename for clearer identification
                if 'rectangle' in sample_file:
                    shape_name = 'rectangle'
                elif 'circle' in sample_file:
                    shape_name = 'circle'
                elif 'triangle' in sample_file:
                    shape_name = 'triangle'
                elif 'ellipse' in sample_file:
                    shape_name = 'ellipse'
                elif 'pentagon' in sample_file:
                    shape_name = 'pentagon'
                else:
                    shape_name = sample_file.replace('.png', '').replace('sample', 'shape')
                
                reference_signatures[shape_name] = normalized_ref
        
        if not reference_signatures:
            return jsonify({'success': False, 'message': 'No reference signatures could be extracted'})
        
        # Calculate similarity measures using Euclidean distance
        similarities = {}
        for ref_name, ref_sig in reference_signatures.items():
            # Ensure both signatures have the same length
            min_len = min(len(normalized_signature), len(ref_sig))
            current_sig = normalized_signature[:min_len]
            reference_sig = ref_sig[:min_len]
            
            # Calculate Euclidean distance according to theory:
            # Formula: sqrt(sum((xi - yi)^2))
            # where xi and yi are corresponding elements of the two signatures
            squared_differences = (current_sig - reference_sig) ** 2
            distance = np.sqrt(np.sum(squared_differences))
            
            similarities[ref_name] = distance
        
        # Sort by similarity (smallest distance = most similar according to theory)
        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1])
        
        # Find the best match (minimum distance indicates highest similarity)
        if sorted_similarities:
            best_match = sorted_similarities[0]
            classification_result = {
                'best_match': best_match[0],
                'distance': best_match[1],
                'all_similarities': sorted_similarities
            }
            
            return jsonify({
                'success': True,
                'message': f'Classification completed. Best match: {best_match[0]}',
                'classification': classification_result
            })
        else:
            return jsonify({'success': False, 'message': 'No valid similarity comparisons could be made'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error in classification: {str(e)}'})

@app.route('/get_file_list')
def get_file_list():
    """Get list of processed files for the file table"""
    try:
        files = []
        sample_images = get_sample_images()
        
        for i, filename in enumerate(sample_images[:5], 1):  # Show 5 files to match sample images
            files.append({
                'no': i,
                'filename': filename
            })
        
        return jsonify({'success': True, 'files': files})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error getting file list: {str(e)}'})

@app.route('/get_similarity_table')
def get_similarity_table():
    """Get similarity comparison table data"""
    try:
        # Return dynamic similarity data based on actual reference signatures
        similarity_data = []
        
        if reference_signatures:
            for i, (ref_name, _) in enumerate(reference_signatures.items(), 1):
                similarity_data.append({
                    'no': i,
                    'reference': ref_name,
                    'similarity': 0.0  # Will be updated during classification
                })
        else:
            # Default data if no classification has been performed yet
            sample_images = get_sample_images()
            for i, sample_file in enumerate(sample_images[:5], 1):
                if 'rectangle' in sample_file:
                    shape_name = 'rectangle'
                elif 'circle' in sample_file:
                    shape_name = 'circle'
                elif 'triangle' in sample_file:
                    shape_name = 'triangle'
                elif 'ellipse' in sample_file:
                    shape_name = 'ellipse'
                elif 'pentagon' in sample_file:
                    shape_name = 'pentagon'
                else:
                    shape_name = f'shape{i}'
                
                similarity_data.append({
                    'no': i,
                    'reference': shape_name,
                    'similarity': 0.0
                })
        
        return jsonify({'success': True, 'similarities': similarity_data})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error getting similarity data: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)