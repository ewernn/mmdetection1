import pandas as pd
import cv2
import os
from tqdm import tqdm
import numpy as np
import json
import argparse

def is_valid_bbox(x1, y1, x2, y2):
    return all(v is not None and v != '' and v != -1.0 and not np.isnan(v) for v in [x1, y1, x2, y2])

def draw_bboxes_coco(image_path, annotations, output_path, image_id):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        return False, []
    height, width = img.shape[:2]

    # Function to draw a single box
    def draw_box(bbox, color, label):
        x, y, w, h = bbox
        if is_valid_bbox(x, y, x+w, y+h):
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.circle(img, (x1, y1), 5, color, -1)
            cv2.circle(img, (x2, y2), 5, color, -1)
            return f"{label} bbox: ({x1}, {y1}, {x2}, {y2})"
        return f"Skipping invalid {label} bbox: ({x}, {y}, {w}, {h})"

    bbox_info = []
    colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0)]  # Red, Blue, Green
    for i, ann in enumerate(annotations):
        bbox_info.append((image_id, os.path.basename(image_path), 
                         draw_box(ann['bbox'], colors[i % len(colors)], f"Box_{i+1}")))

    # Draw the original image name and ID on the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (255, 255, 255)
    font_thickness = 1
    text = f"{os.path.basename(image_path)} (ID: {image_id})"
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = 10
    text_y = height - 10

    cv2.rectangle(img, (text_x, text_y - text_size[1] - 10), 
                 (text_x + text_size[0], text_y), (0, 0, 0), -1)
    cv2.putText(img, text, (text_x, text_y), font, font_scale, font_color, font_thickness)

    cv2.imwrite(output_path, img)
    return True, bbox_info

def main():
    parser = argparse.ArgumentParser(description='Visualize COCO annotations')
    # parser.add_argument('--coco_json', type=str, default="/Users/ewern/Desktop/code/MetronMind/data/cat-data-combined-oct30/train.json", help='Path to COCO JSON file')
    # parser.add_argument('--image_dir', type=str, default="/Users/ewern/Desktop/code/MetronMind/data/cat-data-combined-oct30", help='Path to image directory')
    # parser.add_argument('--output_dir', type=str, default="/Users/ewern/Desktop/code/MetronMind/data/cat-data-combined-oct30-visualized", help='Path to output directory')
    parser.add_argument('--coco_json', type=str, default="/Users/ewern/Desktop/code/MetronMind/data/a_fresh_cat_dataset/val.json", help='Path to COCO JSON file')
    parser.add_argument('--image_dir', type=str, default="/Users/ewern/Desktop/code/MetronMind/data/a_fresh_cat_dataset/images/", help='Path to image directory')
    parser.add_argument('--output_dir', type=str, default="/Users/ewern/Desktop/code/MetronMind/data/a_fresh_cat_dataset/COCO_visualized", help='Path to output directory')
    parser.add_argument('--num_images', type=int, default=-1, help='Number of images to process (-1 for all)')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load COCO JSON
    with open(args.coco_json, 'r') as f:
        coco_data = json.load(f)

    # Create image_id to annotations mapping
    image_to_anns = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_to_anns:
            image_to_anns[image_id] = []
        image_to_anns[image_id].append(ann)

    # Process images
    images = coco_data['images']
    if args.num_images > 0:
        images = images[:args.num_images]

    # Keep track of image names and their counts
    image_counts = {}
    all_bbox_info = []
    processed_count = 0
    skipped_count = 0

    for img_info in tqdm(images, desc="Processing images"):
        image_path = os.path.join(args.image_dir, img_info['file_name'])
        base_name = os.path.basename(img_info['file_name'])
        
        output_filename = f"annotated_id{img_info['id']}_{base_name}"
        output_path = os.path.join(args.output_dir, output_filename)
        
        if base_name not in image_counts:
            image_counts[base_name] = 1
        else:
            image_counts[base_name] += 1

        if not os.path.exists(image_path):
            skipped_count += 1
            continue

        annotations = image_to_anns.get(img_info['id'], [])
        success, bbox_info = draw_bboxes_coco(image_path, annotations, output_path, img_info['id'])
        
        if success:
            processed_count += 1
            all_bbox_info.extend(bbox_info)

    # Print summary information
    print(f"\nProcessing Summary:")
    print(f"Total images processed: {processed_count}")
    print(f"Total images skipped: {skipped_count}")
    print(f"Total images: {len(images)}")

    # Find duplicate image names
    duplicates = {name: count for name, count in image_counts.items() if count > 1}
    if duplicates:
        print("\nDuplicate Images and their Bounding Boxes:")
        for img_id, img_name, bbox_info in all_bbox_info:
            if img_name in duplicates:
                print(f"\nImage: {img_name} (ID: {img_id})")
                print(bbox_info)
        
        print("\nDuplicate Image Names Summary:")
        for name, count in duplicates.items():
            print(f"{name}: {count} occurrences")
    else:
        print("\nNo duplicate image names found.")

if __name__ == "__main__":
    main()