import pandas as pd
import cv2
import os
from tqdm import tqdm

def draw_bboxes(image_path, row, output_path):
    # Read the image
    img = cv2.imread(image_path)
    height, width = img.shape[:2]

    # Function to draw a single box
    def draw_box(x1, y1, x2, y2, color):
        if x1 >= 0 and y1 >= 0 and x2 >= 0 and y2 >= 0:
            x1, y1 = int(x1 * width), int(y1 * height)
            x2, y2 = int(x2 * width), int(y2 * height)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.circle(img, (x1, y1), 5, color, -1)
            cv2.circle(img, (x2, y2), 5, color, -1)

    # Draw first box in red
    draw_box(row['x1'], row['y1'], row['x2'], row['y2'], (0, 0, 255))

    # Draw second box in blue, if it exists
    draw_box(row['x3'], row['y3'], row['x4'], row['y4'], (255, 0, 0))

    # Save the annotated image
    cv2.imwrite(output_path, img)

def main():
    # Read the CSV file
    df = pd.read_csv('/Users/ewern/Downloads/Train-1634/Data.csv')
    df.columns = df.columns.str.strip()
    # Create output directory if it doesn't exist
    output_dir = '/Users/ewern/Downloads/Train-1634-annotated'
    os.makedirs(output_dir, exist_ok=True)

    # Process the first n images
    for _, row in tqdm(df.iterrows(), desc="Processing images"):
        image_path = os.path.join('/Users/ewern/Downloads/Train-1634', row['Image'])
        output_path = os.path.join(output_dir, f"annotated_{row['Image']}")
        draw_bboxes(image_path, row, output_path)


if __name__ == "__main__":
    main()