"""
Utility script to generate printable ArUco markers for table corners.
Run this script to create marker images that you can print.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.marker_detector import generate_corner_markers

if __name__ == "__main__":
    # Output directory
    output_dir = os.path.join(os.path.dirname(__file__), "..", "markers")
    
    print("=" * 50)
    print("ArUco Marker Generator for Table Guidance System")
    print("=" * 50)
    
    generate_corner_markers(output_dir, marker_size=600)
    
    print("\n" + "=" * 50)
    print("INSTRUCTIONS:")
    print("=" * 50)
    print(f"\n1. Find the generated markers in: {os.path.abspath(output_dir)}")
    print("\n2. Print each marker on paper (A5 or A6 size works well)")
    print("\n3. Place markers at your table corners:")
    print("   - Marker ID 0: TOP-LEFT corner")
    print("   - Marker ID 1: TOP-RIGHT corner")
    print("   - Marker ID 2: BOTTOM-RIGHT corner")
    print("   - Marker ID 3: BOTTOM-LEFT corner")
    print("\n4. Make sure markers are flat and visible to the camera")
    print("\n5. The white border around each marker helps with detection")
    print("=" * 50)
