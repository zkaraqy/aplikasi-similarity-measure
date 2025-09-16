from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server deployment
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
import json
from datetime import datetime
import math
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import OpenCV, fallback to PIL if not available
try:
    import cv2
    OPENCV_AVAILABLE = True
    logger.info("OpenCV loaded successfully")
except ImportError:
    OPENCV_AVAILABLE = False
    logger.warning("OpenCV not available, using PIL fallback")

# Set OpenCV to not use GUI
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'

app = Flask(__name__)
app.secret_key = 'similarity_measure_secret_key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['SAMPLE_FOLDER'] = 'static/sample_images'
app.config['PLOTS_FOLDER'] = 'static/plots'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PLOTS_FOLDER'], exist_ok=True)

logger.info(f"Flask app initialized")
logger.info(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
logger.info(f"Sample folder: {app.config['SAMPLE_FOLDER']}")
logger.info(f"Plots folder: {app.config['PLOTS_FOLDER']}")
logger.info(f"OpenCV available: {OPENCV_AVAILABLE}")

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
        if OPENCV_AVAILABLE:
            return extract_with_opencv(image_path)
        else:
            return extract_with_pil(image_path)
    except Exception as e:
        print(f"Error extracting signature from {image_path}: {e}")
        return None

def extract_with_opencv(image_path):
    """Extract signature using OpenCV"""
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
    
    return calculate_signature_from_contour(largest_contour, cx, cy)

def extract_with_pil(image_path):
    """Extract signature using PIL as fallback"""
    from PIL import Image
    
    # Load image with PIL
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img_array = np.array(img)
    
    # Simple thresholding
    threshold = np.mean(img_array)
    binary = (img_array > threshold).astype(np.uint8) * 255
    
    # Find contour points (simplified edge detection)
    edges = []
    height, width = binary.shape
    
    for y in range(1, height-1):
        for x in range(1, width-1):
            if binary[y, x] == 255:  # White pixel (object)
                # Check if it's an edge pixel
                neighbors = binary[y-1:y+2, x-1:x+2]
                if np.any(neighbors == 0):  # Has black neighbors
                    edges.append([[[x, y]]])  # Format like OpenCV contour
    
    if not edges:
        return None
    
    # Calculate centroid
    edges_array = np.array([point[0][0] for point in edges])
    cx = np.mean(edges_array[:, 0])
    cy = np.mean(edges_array[:, 1])
    
    return calculate_signature_from_contour(edges, cx, cy)

def calculate_signature_from_contour(contour, cx, cy):
    """Calculate angle signature from contour points and centroid"""
    signature = []
    angles = np.arange(0, 360, 1)  # 360 points for 0-359 degrees
    
    for angle in angles:
        # Find distances for this angle
        distances = []
        for point in contour:
            if OPENCV_AVAILABLE:
                px, py = point[0]
            else:
                px, py = point[0][0]  # PIL format
            
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

def get_sample_images():
    """Get list of sample images"""
    sample_path = app.config['SAMPLE_FOLDER']
    if not os.path.exists(sample_path):
        os.makedirs(sample_path, exist_ok=True)
    
    images = []
    for filename in os.listdir(sample_path):
        if allowed_file(filename):
            images.append(filename)
    
    # If no sample images, generate them
    if not images:
        try:
            generate_sample_images_embedded()
            # Recheck after generation
            for filename in os.listdir(sample_path):
                if allowed_file(filename):
                    images.append(filename)
        except Exception as e:
            print(f"Could not generate sample images: {e}")
    
    return sorted(images)

def generate_sample_images_embedded():
    """Generate sample images embedded in the app"""
    try:
        output_dir = app.config['SAMPLE_FOLDER']
        os.makedirs(output_dir, exist_ok=True)
        
        # Image dimensions
        width, height = 400, 400
        object_color = 255  # White objects
        
        if OPENCV_AVAILABLE:
            # Sample 1: Rectangle
            img1 = np.zeros((height, width), dtype=np.uint8)
            cv2.rectangle(img1, (150, 120), (250, 200), object_color, -1)
            cv2.imwrite(os.path.join(output_dir, "sample1_rectangle.png"), img1)
            
            # Sample 2: Circle
            img2 = np.zeros((height, width), dtype=np.uint8)
            cv2.circle(img2, (200, 160), 60, object_color, -1)
            cv2.imwrite(os.path.join(output_dir, "sample2_circle.png"), img2)
            
            # Sample 3: Triangle
            img3 = np.zeros((height, width), dtype=np.uint8)
            triangle_points = np.array([[200, 100], [150, 200], [250, 200]], np.int32)
            cv2.fillPoly(img3, [triangle_points], object_color)
            cv2.imwrite(os.path.join(output_dir, "sample3_triangle.png"), img3)
            
            # Sample 4: Ellipse
            img4 = np.zeros((height, width), dtype=np.uint8)
            cv2.ellipse(img4, (200, 160), (80, 50), 45, 0, 360, object_color, -1)
            cv2.imwrite(os.path.join(output_dir, "sample4_ellipse.png"), img4)
            
            # Sample 5: Pentagon
            img5 = np.zeros((height, width), dtype=np.uint8)
            center = (200, 160)
            radius = 70
            pentagon_points = []
            for i in range(5):
                angle = i * 2 * np.pi / 5 - np.pi / 2
                x = int(center[0] + radius * np.cos(angle))
                y = int(center[1] + radius * np.sin(angle))
                pentagon_points.append([x, y])
            pentagon_points = np.array(pentagon_points, np.int32)
            cv2.fillPoly(img5, [pentagon_points], object_color)
            cv2.imwrite(os.path.join(output_dir, "sample5_pentagon.png"), img5)
        else:
            # PIL fallback
            from PIL import Image, ImageDraw
            
            # Sample 1: Rectangle
            img1 = Image.new('L', (width, height), 0)
            draw = ImageDraw.Draw(img1)
            draw.rectangle([150, 120, 250, 200], fill=255)
            img1.save(os.path.join(output_dir, "sample1_rectangle.png"))
            
            # Sample 2: Circle
            img2 = Image.new('L', (width, height), 0)
            draw = ImageDraw.Draw(img2)
            draw.ellipse([140, 100, 260, 220], fill=255)
            img2.save(os.path.join(output_dir, "sample2_circle.png"))
            
            # Sample 3: Triangle
            img3 = Image.new('L', (width, height), 0)
            draw = ImageDraw.Draw(img3)
            draw.polygon([(200, 100), (150, 200), (250, 200)], fill=255)
            img3.save(os.path.join(output_dir, "sample3_triangle.png"))
            
            # Sample 4: Ellipse
            img4 = Image.new('L', (width, height), 0)
            draw = ImageDraw.Draw(img4)
            draw.ellipse([120, 110, 280, 210], fill=255)
            img4.save(os.path.join(output_dir, "sample4_ellipse.png"))
            
            # Sample 5: Pentagon (simplified as circle for PIL)
            img5 = Image.new('L', (width, height), 0)
            draw = ImageDraw.Draw(img5)
            # Approximate pentagon with polygon
            points = []
            center = (200, 160)
            radius = 70
            for i in range(5):
                angle = i * 2 * math.pi / 5 - math.pi / 2
                x = int(center[0] + radius * math.cos(angle))
                y = int(center[1] + radius * math.sin(angle))
                points.append((x, y))
            draw.polygon(points, fill=255)
            img5.save(os.path.join(output_dir, "sample5_pentagon.png"))
        
        print("Sample images generated successfully!")
        
    except Exception as e:
        print(f"Error generating sample images: {e}")
        raise

@app.route('/')
def index():
    """Main application page"""
    try:
        sample_images = get_sample_images()
        return render_template('index.html', sample_images=sample_images)
    except Exception as e:
        print(f"Error in index route: {e}")
        return f"Application Error: {str(e)}", 500

@app.route('/health')
def health_check():
    """Health check endpoint for Railway"""
    try:
        # Check if sample images exist
        sample_images = get_sample_images()
        
        # Check if directories exist
        dirs_exist = all([
            os.path.exists(app.config['UPLOAD_FOLDER']),
            os.path.exists(app.config['PLOTS_FOLDER']),
            os.path.exists(app.config['SAMPLE_FOLDER'])
        ])
        
        logger.info(f"Health check - Sample images: {len(sample_images)}, Directories OK: {dirs_exist}, OpenCV: {OPENCV_AVAILABLE}")
        
        return jsonify({
            'status': 'healthy',
            'sample_images_count': len(sample_images),
            'directories_ready': dirs_exist,
            'opencv_available': OPENCV_AVAILABLE
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

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
        
        if OPENCV_AVAILABLE:
            # OpenCV implementation
            gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = np.ones((3,3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            segmented_image = binary
        else:
            # PIL fallback
            from PIL import Image
            if len(current_image.shape) == 3:
                gray = np.mean(current_image, axis=2).astype(np.uint8)
            else:
                gray = current_image
            
            threshold = np.mean(gray)
            segmented_image = (gray > threshold).astype(np.uint8) * 255
        
        # Save segmented image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        seg_filename = f"segmented_{timestamp}.png"
        seg_path = os.path.join(app.config['UPLOAD_FOLDER'], seg_filename)
        
        if OPENCV_AVAILABLE:
            cv2.imwrite(seg_path, segmented_image)
        else:
            from PIL import Image
            Image.fromarray(segmented_image).save(seg_path)
        
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
    # Get port from environment variable (Railway sets this)
    port = int(os.environ.get('PORT', 5000))
    
    logger.info(f"Starting application on port {port}")
    
    # Create necessary directories if they don't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PLOTS_FOLDER'], exist_ok=True)
    os.makedirs(app.config['SAMPLE_FOLDER'], exist_ok=True)
    
    logger.info("Directories created/verified")
    
    # Generate sample images if they don't exist
    sample_images = get_sample_images()
    if not sample_images:
        logger.info("No sample images found, generating...")
        try:
            exec(open('generate_samples.py').read())
            logger.info("Sample images generated successfully")
        except Exception as e:
            logger.error(f"Could not generate sample images: {e}")
    else:
        logger.info(f"Found {len(sample_images)} sample images")
    
    logger.info(f"Starting Flask app on port {port}")
    app.run(debug=False, host='0.0.0.0', port=port)