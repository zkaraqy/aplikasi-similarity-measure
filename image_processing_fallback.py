"""
Alternative image processing module using PIL/Pillow for environments where OpenCV is problematic
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import os

def extract_angle_signature_with_pil(image_path):
    """Extract angle signature using PIL as fallback"""
    try:
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
                        edges.append((x, y))
        
        if not edges:
            return None
        
        # Calculate centroid
        edges = np.array(edges)
        cx = np.mean(edges[:, 0])
        cy = np.mean(edges[:, 1])
        
        # Calculate angle signature
        signature = []
        for angle in range(360):
            angle_rad = np.radians(angle)
            distances = []
            
            for x, y in edges:
                point_angle = np.degrees(np.arctan2(y - cy, x - cx))
                if point_angle < 0:
                    point_angle += 360
                
                if abs(point_angle - angle) < 2:
                    distance = np.sqrt((x - cx)**2 + (y - cy)**2)
                    distances.append(distance)
            
            if distances:
                signature.append(max(distances))
            else:
                signature.append(0)
        
        # Smooth signature
        signature = np.array(signature)
        for i in range(len(signature)):
            if signature[i] == 0:
                # Simple interpolation
                left_val = 0
                right_val = 0
                for j in range(1, 10):
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
        print(f"PIL fallback error: {e}")
        return None