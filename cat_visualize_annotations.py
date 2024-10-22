import pandas as pd
import cv2
import os
from tqdm import tqdm
import numpy as np

def is_valid_bbox(x1, y1, x2, y2):
    return all(v is not None and v != '' and v != -1.0 and not np.isnan(v) for v in [x1, y1, x2, y2])

def draw_bboxes(image_path, row, output_path):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Unable to read image {image_path}. Skipping this image.")
        return False
    height, width = img.shape[:2]

    # Function to draw a single box
    def draw_box(x1, y1, x2, y2, color, label):
        if is_valid_bbox(x1, y1, x2, y2):
            x1, y1 = int(x1 * width), int(y1 * height)
            x2, y2 = int(x2 * width), int(y2 * height)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.circle(img, (x1, y1), 5, color, -1)
            cv2.circle(img, (x2, y2), 5, color, -1)
            print(f"{label} bbox: ({x1}, {y1}, {x2}, {y2})")
        else:
            print(f"Skipping invalid {label} bbox: ({x1}, {y1}, {x2}, {y2})")

    print(f"Drawing bboxes for image: {os.path.basename(image_path)}")
    
    # Draw first box in red
    draw_box(row['x1'], row['y1'], row['x2'], row['y2'], (0, 0, 255), "First")

    # Draw second box in blue, if it exists
    if all(col in row for col in ['x3', 'y3', 'x4', 'y4']):
        draw_box(row['x3'], row['y3'], row['x4'], row['y4'], (255, 0, 0), "Second")

    # Draw the original image name on the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (255, 255, 255)  # White color
    font_thickness = 1
    text = os.path.basename(image_path)
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = 10
    text_y = height - 10  # 10 pixels from the bottom

    # Draw a semi-transparent background for the text
    cv2.rectangle(img, (text_x, text_y - text_size[1] - 10), (text_x + text_size[0], text_y), (0, 0, 0), -1)
    cv2.putText(img, text, (text_x, text_y), font, font_scale, font_color, font_thickness)

    # Save the annotated image
    cv2.imwrite(output_path, img)
    return True

def main():
    # Read the CSV file
    df = pd.read_csv('/Users/ewern/Desktop/code/MetronMind/data/DataCatSep26/Data_only2.csv')
    df.columns = df.columns.str.strip()
    
    # Create output directory if it doesn't exist
    output_dir = '/Users/ewern/Desktop/code/MetronMind/data/DataCatSep26-Dataonly2-attempt2'
    os.makedirs(output_dir, exist_ok=True)

    # Process all images
    i = 0
    for _, row in tqdm(df.iterrows(), desc="Processing images"):
        if i > 20:
            break
        i += 1
        image_path = os.path.join('/Users/ewern/Desktop/code/MetronMind/data/DataCatSep26', row['Image'])
        output_path = os.path.join(output_dir, f"annotated_{row['Image']}")
        
        if not os.path.exists(image_path):
            print(f"Warning: Image file not found: {image_path}")
            continue

        draw_bboxes(image_path, row, output_path)
        print()  # Add a blank line for readability between images

if __name__ == "__main__":
    main()