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
    
    # Sample 1: Rectangle
    img1 = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(img1, (150, 120), (250, 200), object_color, -1)
    cv2.imwrite(os.path.join(output_dir, "sample1_rectangle.png"), img1)
    
    # Sample 2: Circle
    img2 = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.circle(img2, (200, 160), 60, object_color, -1)
    cv2.imwrite(os.path.join(output_dir, "sample2_circle.png"), img2)
    
    # Sample 3: Triangle
    img3 = np.zeros((height, width, 3), dtype=np.uint8)
    triangle_points = np.array([[200, 100], [150, 200], [250, 200]], np.int32)
    cv2.fillPoly(img3, [triangle_points], object_color)
    cv2.imwrite(os.path.join(output_dir, "sample3_triangle.png"), img3)
    
    # Sample 4: Ellipse
    img4 = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.ellipse(img4, (200, 160), (80, 50), 45, 0, 360, object_color, -1)
    cv2.imwrite(os.path.join(output_dir, "sample4_ellipse.png"), img4)
    
    # Sample 5: Pentagon
    img5 = np.zeros((height, width, 3), dtype=np.uint8)
    center = (200, 160)
    radius = 70
    pentagon_points = []
    for i in range(5):
        angle = i * 2 * np.pi / 5 - np.pi / 2  # Start from top
        x = int(center[0] + radius * np.cos(angle))
        y = int(center[1] + radius * np.sin(angle))
        pentagon_points.append([x, y])
    pentagon_points = np.array(pentagon_points, np.int32)
    cv2.fillPoly(img5, [pentagon_points], object_color)
    cv2.imwrite(os.path.join(output_dir, "sample5_pentagon.png"), img5)
    
    print("Sample images created successfully!")
    print("Generated files:")
    for i in range(1, 6):
        shapes = ["rectangle", "circle", "triangle", "ellipse", "pentagon"]
        filename = f"sample{i}_{shapes[i-1]}.png"
        print(f"  - {filename}")

if __name__ == "__main__":
    create_sample_images()