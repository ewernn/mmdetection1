import cv2
import os

def draw_bboxes_on_image():
    # Image path
    image_path = "/Users/ewern/Desktop/code/MetronMind/data/cat-data-combined-oct30/Im1342.tif"
    
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    # First set of boxes (ID: 1342) in red
    cv2.rectangle(img, (564, 231), (727, 439), (0, 0, 255), 2)  # Box_1
    cv2.rectangle(img, (247, 141), (435, 356), (0, 0, 255), 2)  # Box_2
    
    # Second set of boxes (ID: 153) in blue
    cv2.rectangle(img, (234, 263), (383, 474), (255, 0, 0), 2)  # Box_1
    cv2.rectangle(img, (575, 347), (725, 558), (255, 0, 0), 2)  # Box_2
    
    # Add text to identify the sets
    cv2.putText(img, "ID: 1342 (Red)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(img, "ID: 153 (Blue)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # Save the image
    output_path = "/Users/ewern/Desktop/code/MetronMind/data/cat-data-combined-oct30-Im1342_comparison.tif"
    cv2.imwrite(output_path, img)
    print(f"Image saved to {output_path}")

if __name__ == "__main__":
    draw_bboxes_on_image()