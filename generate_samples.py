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
    
    # Sample 2: Jam Tangan (Watch - circular)
    img2 = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.circle(img2, (200, 160), 60, object_color, -1)
    cv2.circle(img2, (200, 160), 45, (100, 100, 100), -1)  # Inner circle
    cv2.imwrite(os.path.join(output_dir, "jam_tangan.png"), img2)
    
    # Sample 3: Piring (Plate - circular with rim)
    img3 = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.circle(img3, (200, 160), 70, object_color, -1)
    cv2.circle(img3, (200, 160), 50, (150, 150, 150), -1)  # Inner plate
    cv2.imwrite(os.path.join(output_dir, "piring.png"), img3)
    
    # Sample 4: Uang Koin (Coin - circular)
    img4 = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.circle(img4, (200, 160), 50, object_color, -1)
    cv2.imwrite(os.path.join(output_dir, "uang_koin.png"), img4)
    
    # Sample 5: Wadai (Container - elliptical)
    img5 = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.ellipse(img5, (200, 160), (80, 50), 0, 0, 360, object_color, -1)
    cv2.imwrite(os.path.join(output_dir, "wadai.png"), img5)
    
    print("Sample images created successfully!")
    print("Generated files:")
    files = ["gantungan_kunci.png", "jam_tangan.png", "piring.png", "uang_koin.png", "wadai.png"]
    descriptions = ["Gantungan Kunci", "Jam Tangan", "Piring", "Uang Koin", "Wadai"]
    
    for filename, desc in zip(files, descriptions):
        print(f"  - {filename} ({desc})")
    
    return True

if __name__ == "__main__":
    create_sample_images()