"""
Color Calibration Tool
======================
Use this tool to find the correct HSV color ranges for your objects.
Adjust the sliders until only your target object is highlighted.

Controls:
- Adjust H (Hue), S (Saturation), V (Value) sliders
- Press 'p' to print current values (copy to procedure.json)
- Press 'q' to quit
"""

import cv2
import numpy as np


def main():
    print("=" * 50)
    print("Color Calibration Tool")
    print("=" * 50)
    print("\nThis tool helps you find HSV color ranges for object detection.")
    print("\nInstructions:")
    print("1. Place your colored object in front of the camera")
    print("2. Adjust the sliders until only your object appears white in the mask")
    print("3. Press 'p' to print the values")
    print("4. Copy the values to config/procedure.json")
    print("\nPress 'q' to quit")
    print("=" * 50)
    
    # Create window and trackbars
    cv2.namedWindow("Color Calibrator")
    
    # HSV ranges for different colors (starting points)
    color_presets = {
        '1': ('Red', [0, 100, 100], [10, 255, 255]),
        '2': ('Blue', [100, 100, 100], [130, 255, 255]),
        '3': ('Green', [35, 100, 100], [85, 255, 255]),
        '4': ('Yellow', [20, 100, 100], [35, 255, 255]),
        '5': ('Orange', [10, 100, 100], [25, 255, 255]),
    }
    
    # Initialize trackbars
    cv2.createTrackbar("H Low", "Color Calibrator", 0, 180, lambda x: None)
    cv2.createTrackbar("H High", "Color Calibrator", 180, 180, lambda x: None)
    cv2.createTrackbar("S Low", "Color Calibrator", 100, 255, lambda x: None)
    cv2.createTrackbar("S High", "Color Calibrator", 255, 255, lambda x: None)
    cv2.createTrackbar("V Low", "Color Calibrator", 100, 255, lambda x: None)
    cv2.createTrackbar("V High", "Color Calibrator", 255, 255, lambda x: None)
    
    cap = cv2.VideoCapture(0)
    
    print("\nPresets (press number key to load):")
    for key, (name, _, _) in color_presets.items():
        print(f"  {key}: {name}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Get trackbar values
        h_low = cv2.getTrackbarPos("H Low", "Color Calibrator")
        h_high = cv2.getTrackbarPos("H High", "Color Calibrator")
        s_low = cv2.getTrackbarPos("S Low", "Color Calibrator")
        s_high = cv2.getTrackbarPos("S High", "Color Calibrator")
        v_low = cv2.getTrackbarPos("V Low", "Color Calibrator")
        v_high = cv2.getTrackbarPos("V High", "Color Calibrator")
        
        lower = np.array([h_low, s_low, v_low])
        upper = np.array([h_high, s_high, v_high])
        
        # Create mask
        mask = cv2.inRange(hsv, lower, upper)
        
        # Apply morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Apply mask to frame
        result = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Add text overlay
        info_text = f"Lower: [{h_low}, {s_low}, {v_low}]  Upper: [{h_high}, {s_high}, {v_high}]"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'p' to print, 'q' to quit", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Combine views
        mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        top_row = np.hstack([frame, mask_color])
        bottom_row = np.hstack([result, np.zeros_like(result)])
        combined = np.vstack([top_row, bottom_row])
        
        # Resize if too large
        h, w = combined.shape[:2]
        if w > 1280:
            scale = 1280 / w
            combined = cv2.resize(combined, (int(w * scale), int(h * scale)))
        
        cv2.imshow("Color Calibrator", combined)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('p'):
            print("\n" + "=" * 40)
            print("Copy these values to procedure.json:")
            print("=" * 40)
            print(f'"color_lower": [{h_low}, {s_low}, {v_low}],')
            print(f'"color_upper": [{h_high}, {s_high}, {v_high}],')
            print("=" * 40)
        elif chr(key) in color_presets:
            name, lower, upper = color_presets[chr(key)]
            print(f"\nLoading preset: {name}")
            cv2.setTrackbarPos("H Low", "Color Calibrator", lower[0])
            cv2.setTrackbarPos("H High", "Color Calibrator", upper[0])
            cv2.setTrackbarPos("S Low", "Color Calibrator", lower[1])
            cv2.setTrackbarPos("S High", "Color Calibrator", upper[1])
            cv2.setTrackbarPos("V Low", "Color Calibrator", lower[2])
            cv2.setTrackbarPos("V High", "Color Calibrator", upper[2])
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
