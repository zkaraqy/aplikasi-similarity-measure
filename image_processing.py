"""
Image Processing Module for Similarity Measure Application
Contains algorithms for segmentation, angle signature calculation, 
normalization, and similarity measurement.
"""

import cv2
import numpy as np
import math
from typing import Tuple, List, Optional

class ImageProcessor:
    """Main class for image processing operations"""
    
    def __init__(self):
        self.current_image = None
        self.segmented_image = None
        self.angle_signature = None
        self.normalized_signature = None
        self.centroid = None
        self.contour = None
    
    def load_image(self, image_path: str) -> bool:
        """Load image from file path"""
        try:
            self.current_image = cv2.imread(image_path)
            return self.current_image is not None
        except Exception:
            return False
    
    def segment_image(self, method: str = 'otsu') -> Optional[np.ndarray]:
        """
        Perform image segmentation using various methods
        
        Args:
            method: Segmentation method ('otsu', 'adaptive', 'manual')
        
        Returns:
            Binary segmented image or None if failed
        """
        if self.current_image is None:
            return None
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            
            if method == 'otsu':
                # Otsu's thresholding
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            elif method == 'adaptive':
                # Adaptive thresholding
                binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 11, 2)
            else:
                # Manual thresholding
                _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            # Apply morphological operations to clean up
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            # Remove small noise
            binary = cv2.medianBlur(binary, 5)
            
            self.segmented_image = binary
            return binary
            
        except Exception as e:
            print(f"Segmentation error: {e}")
            return None
    
    def find_main_contour(self) -> Optional[np.ndarray]:
        """Find the main object contour from segmented image"""
        if self.segmented_image is None:
            return None
        
        try:
            # Find contours
            contours, _ = cv2.findContours(self.segmented_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            if not contours:
                return None
            
            # Get the largest contour (assuming it's the main object)
            self.contour = max(contours, key=cv2.contourArea)
            
            # Calculate centroid
            M = cv2.moments(self.contour)
            if M['m00'] == 0:
                return None
            
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            self.centroid = (cx, cy)
            
            return self.contour
            
        except Exception as e:
            print(f"Contour detection error: {e}")
            return None
    
    def calculate_angle_signature(self, num_points: int = 360) -> Optional[np.ndarray]:
        """
        Calculate angle distance signature for the main contour
        
        Args:
            num_points: Number of angle points (default 360 for each degree)
        
        Returns:
            Angle signature array or None if failed
        """
        if self.contour is None or self.centroid is None:
            if not self.find_main_contour():
                return None
        
        try:
            cx, cy = self.centroid
            signature = []
            
            # Calculate signature for each angle
            for angle_deg in range(num_points):
                # Convert to radians
                angle_rad = np.radians(angle_deg)
                
                # Find maximum distance at this angle
                max_distance = 0
                
                # Check all contour points
                for point in self.contour:
                    px, py = point[0]
                    
                    # Calculate angle from centroid to this point
                    point_angle = np.arctan2(py - cy, px - cx)
                    point_angle_deg = np.degrees(point_angle)
                    if point_angle_deg < 0:
                        point_angle_deg += 360
                    
                    # Check if this point is at the current angle (with tolerance)
                    angle_diff = abs(point_angle_deg - angle_deg)
                    if angle_diff < 1.0 or angle_diff > 359.0:  # Handle wrap-around
                        distance = np.sqrt((px - cx)**2 + (py - cy)**2)
                        max_distance = max(max_distance, distance)
                
                # If no point found at this angle, interpolate
                if max_distance == 0:
                    # Use ray casting to find boundary intersection
                    ray_end_x = cx + 1000 * np.cos(angle_rad)
                    ray_end_y = cy + 1000 * np.sin(angle_rad)
                    
                    # Find intersection with contour (simplified approach)
                    min_dist_to_contour = float('inf')
                    for point in self.contour:
                        px, py = point[0]
                        # Check if point is roughly on the ray direction
                        point_angle = np.arctan2(py - cy, px - cx)
                        if abs(angle_rad - point_angle) < 0.1:
                            distance = np.sqrt((px - cx)**2 + (py - cy)**2)
                            min_dist_to_contour = min(min_dist_to_contour, distance)
                    
                    if min_dist_to_contour != float('inf'):
                        max_distance = min_dist_to_contour
                
                signature.append(max_distance)
            
            self.angle_signature = np.array(signature)
            return self.angle_signature
            
        except Exception as e:
            print(f"Angle signature calculation error: {e}")
            return None
    
    def normalize_signature(self, method: str = 'minmax') -> Optional[np.ndarray]:
        """
        Normalize the angle signature
        
        Args:
            method: Normalization method ('minmax', 'zscore', 'unit')
        
        Returns:
            Normalized signature or None if failed
        """
        if self.angle_signature is None:
            return None
        
        try:
            if method == 'minmax':
                # Min-max normalization to [0, 1]
                min_val = np.min(self.angle_signature)
                max_val = np.max(self.angle_signature)
                
                if max_val == min_val:
                    self.normalized_signature = np.zeros_like(self.angle_signature)
                else:
                    self.normalized_signature = (self.angle_signature - min_val) / (max_val - min_val)
                    
            elif method == 'zscore':
                # Z-score normalization
                mean_val = np.mean(self.angle_signature)
                std_val = np.std(self.angle_signature)
                
                if std_val == 0:
                    self.normalized_signature = np.zeros_like(self.angle_signature)
                else:
                    self.normalized_signature = (self.angle_signature - mean_val) / std_val
                    
            elif method == 'unit':
                # Unit vector normalization
                norm = np.linalg.norm(self.angle_signature)
                if norm == 0:
                    self.normalized_signature = np.zeros_like(self.angle_signature)
                else:
                    self.normalized_signature = self.angle_signature / norm
            
            return self.normalized_signature
            
        except Exception as e:
            print(f"Normalization error: {e}")
            return None
    
    def calculate_similarity(self, reference_signature: np.ndarray, 
                           method: str = 'euclidean') -> float:
        """
        Calculate similarity between current and reference signature
        
        Args:
            reference_signature: Reference signature to compare against
            method: Similarity method ('euclidean', 'cosine', 'correlation')
        
        Returns:
            Similarity measure (lower is more similar for distance measures)
        """
        if self.normalized_signature is None:
            return float('inf')
        
        try:
            if method == 'euclidean':
                # Euclidean distance
                return np.sqrt(np.sum((self.normalized_signature - reference_signature) ** 2))
                
            elif method == 'cosine':
                # Cosine similarity (converted to distance)
                dot_product = np.dot(self.normalized_signature, reference_signature)
                norm_a = np.linalg.norm(self.normalized_signature)
                norm_b = np.linalg.norm(reference_signature)
                
                if norm_a == 0 or norm_b == 0:
                    return 1.0  # Maximum distance
                
                cosine_sim = dot_product / (norm_a * norm_b)
                return 1.0 - cosine_sim  # Convert to distance
                
            elif method == 'correlation':
                # Pearson correlation (converted to distance)
                correlation = np.corrcoef(self.normalized_signature, reference_signature)[0, 1]
                if np.isnan(correlation):
                    return 1.0
                return 1.0 - abs(correlation)  # Convert to distance
                
            else:
                return float('inf')
                
        except Exception as e:
            print(f"Similarity calculation error: {e}")
            return float('inf')

class ReferenceDatabase:
    """Class to manage reference signatures database"""
    
    def __init__(self):
        self.references = {}
    
    def add_reference(self, name: str, signature: np.ndarray, metadata: dict = None):
        """Add a reference signature to the database"""
        self.references[name] = {
            'signature': signature,
            'metadata': metadata or {}
        }
    
    def get_reference(self, name: str) -> Optional[np.ndarray]:
        """Get a reference signature by name"""
        if name in self.references:
            return self.references[name]['signature']
        return None
    
    def get_all_references(self) -> dict:
        """Get all reference signatures"""
        return {name: data['signature'] for name, data in self.references.items()}
    
    def generate_sample_references(self, num_samples: int = 9):
        """Generate sample reference signatures for testing"""
        for i in range(1, num_samples + 1):
            angles = np.arange(0, 360, 1)
            
            if i <= 3:  # Rectangle-like objects
                signature = 0.5 + 0.3 * np.sin(4 * np.radians(angles)) + 0.1 * np.random.random(360)
            elif i <= 6:  # Circular-like objects
                signature = 0.6 + 0.2 * np.sin(2 * np.radians(angles)) + 0.1 * np.random.random(360)
            else:  # Triangle-like objects
                signature = 0.4 + 0.4 * np.sin(3 * np.radians(angles)) + 0.1 * np.random.random(360)
            
            # Normalize
            signature = (signature - np.min(signature)) / (np.max(signature) - np.min(signature))
            
            self.add_reference(f'objek {i}', signature, {'category': 'sample', 'id': i})

def classify_object(processor: ImageProcessor, database: ReferenceDatabase, 
                   method: str = 'euclidean') -> dict:
    """
    Classify an object based on its signature similarity to reference database
    
    Args:
        processor: ImageProcessor instance with calculated normalized signature
        database: ReferenceDatabase with reference signatures
        method: Similarity calculation method
    
    Returns:
        Classification results dictionary
    """
    if processor.normalized_signature is None:
        return {'error': 'No normalized signature available'}
    
    similarities = {}
    references = database.get_all_references()
    
    for ref_name, ref_signature in references.items():
        similarity = processor.calculate_similarity(ref_signature, method)
        similarities[ref_name] = similarity
    
    # Sort by similarity (lowest distance = most similar)
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1])
    
    if sorted_similarities:
        best_match = sorted_similarities[0]
        return {
            'best_match': best_match[0],
            'best_distance': best_match[1],
            'all_similarities': sorted_similarities,
            'classification_confidence': 1.0 / (1.0 + best_match[1])  # Higher is better
        }
    else:
        return {'error': 'No reference signatures available'}