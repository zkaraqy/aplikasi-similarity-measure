"""
Script to generate sample images for testing the similarity measure application.
Creates images with different geometric shapes for angle signature analysis.
"""

import cv2
import numpy as np
import os

def create_sample_images():
    """Generate sample images with different shapes"""
    
    # Create sample_images directory if it doesn't exist
    output_dir = "static/sample_images"
    os.makedirs(output_dir, exist_ok=True)
    
    # Image dimensions
    width, height = 400, 400
    background_color = (0, 0, 0)  # Black background
    object_color = (255, 255, 255)  # White objects
    
    # Sample 1: Gantungan Kunci (Key chain - rectangular with hole)
    img1 = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(img1, (150, 120), (250, 200), object_color, -1)
    cv2.circle(img1, (200, 140), 15, (0, 0, 0), -1)  # Hole for key chain
    cv2.imwrite(os.path.join(output_dir, "gantungan_kunci.png"), img1)
    
    # Sample 2: Gunting (Scissors - two overlapping ovals)
    img2 = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.ellipse(img2, (180, 150), (20, 60), 45, 0, 360, object_color, -1)  # Left blade
    cv2.ellipse(img2, (220, 170), (20, 60), -45, 0, 360, object_color, -1)  # Right blade
    cv2.rectangle(img2, (190, 200), (210, 250), object_color, -1)  # Handle
    cv2.imwrite(os.path.join(output_dir, "gunting.png"), img2)
    
    # Sample 3: Piring (Plate - circular with rim)
    img3 = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.circle(img3, (200, 160), 70, object_color, -1)
    cv2.circle(img3, (200, 160), 50, (150, 150, 150), -1)  # Inner plate
    cv2.imwrite(os.path.join(output_dir, "piring.png"), img3)
    
    # Sample 4: Sisir (Comb - rectangular with teeth)
    img4 = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(img4, (150, 140), (250, 180), object_color, -1)  # Main body
    # Add comb teeth
    for x in range(155, 245, 10):
        cv2.rectangle(img4, (x, 120), (x+5, 140), object_color, -1)
    cv2.imwrite(os.path.join(output_dir, "sisir.png"), img4)
    
    # Sample 5: Wadai (Container - elliptical)
    img5 = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.ellipse(img5, (200, 160), (80, 50), 0, 0, 360, object_color, -1)
    cv2.imwrite(os.path.join(output_dir, "wadai.png"), img5)
    
    print("Sample images created successfully!")
    print("Generated files:")
    files = ["gantungan_kunci.png", "gunting.png", "piring.png", "sisir.png", "wadai.png"]
    descriptions = ["Gantungan Kunci", "Gunting", "Piring", "Sisir", "Wadai"]
    
    for filename, desc in zip(files, descriptions):
        print(f"  - {filename} ({desc})")
    
    return True

if __name__ == "__main__":
    create_sample_images()