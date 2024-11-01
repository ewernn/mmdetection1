# path 1: /Users/ewern/Desktop/code/MetronMind/data/cat_kidney_dataset_csv_filtered/Data.csv
# path 2: /Users/ewern/Desktop/code/MetronMind/data/DataCatSep26/Data_only2.csv
# data format: Image, x1, y1, x2, y2, x3, y3, x4, y4 (x1, y1, x2, y2 are the first box, x3, y3, x4, y4 are the second box)
# image names overlap some, so we need to make a new dataset with images and a csv, starting at Im0.tif and going up
# we are combining two datasets into one
import pandas as pd
import shutil
import os
from tqdm import tqdm

def is_valid_coordinates(row):
    # Check if any coordinate is None, empty, or -1.0
    coords = [row['x1'], row['y1'], row['x2'], row['y2'], 
             row['x3'], row['y3'], row['x4'], row['y4']]
    return all(coord is not None and coord != '' and coord != -1.0 and pd.notna(coord) for coord in coords)

def combine_datasets():
    # Define paths
    path1 = "/Users/ewern/Desktop/code/MetronMind/data/cat_kidney_dataset_csv_filtered/Data.csv"
    path2 = "/Users/ewern/Desktop/code/MetronMind/data/DataCatSep26/Data_only2.csv"
    
    # Define image source directories
    img_dir1 = os.path.dirname(path1)
    img_dir2 = os.path.dirname(path2)
    
    # Create output directories
    output_dir = "/Users/ewern/Desktop/code/MetronMind/data/combined_dataset"
    output_img_dir = os.path.join(output_dir, "images")
    os.makedirs(output_img_dir, exist_ok=True)
    
    # Read CSVs
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    
    # Initialize new dataframe for combined dataset
    combined_data = []
    image_counter = 0
    
    # Process first dataset
    print("Processing first dataset...")
    skipped_count1 = 0
    for _, row in tqdm(df1.iterrows()):
        if not is_valid_coordinates(row):
            skipped_count1 += 1
            continue
            
        old_img_path = os.path.join(img_dir1, row['Image'])
        new_img_name = f"Im{image_counter}.tif"
        new_img_path = os.path.join(output_img_dir, new_img_name)
        
        if os.path.exists(old_img_path):
            shutil.copy2(old_img_path, new_img_path)
            combined_data.append({
                'Image': new_img_name,
                'x1': row['x1'],
                'y1': row['y1'],
                'x2': row['x2'],
                'y2': row['y2'],
                'x3': row['x3'],
                'y3': row['y3'],
                'x4': row['x4'],
                'y4': row['y4'],
                'original_name': row['Image'],
                'source': 'dataset1'
            })
            image_counter += 1
        else:
            print(f"Warning: Image not found - {old_img_path}")
    
    # Process second dataset
    print("Processing second dataset...")
    skipped_count2 = 0
    for _, row in tqdm(df2.iterrows()):
        if not is_valid_coordinates(row):
            skipped_count2 += 1
            continue
            
        old_img_path = os.path.join(img_dir2, row['Image'])
        new_img_name = f"Im{image_counter}.tif"
        new_img_path = os.path.join(output_img_dir, new_img_name)
        
        if os.path.exists(old_img_path):
            shutil.copy2(old_img_path, new_img_path)
            combined_data.append({
                'Image': new_img_name,
                'x1': row['x1'],
                'y1': row['y1'],
                'x2': row['x2'],
                'y2': row['y2'],
                'x3': row['x3'],
                'y3': row['y3'],
                'x4': row['x4'],
                'y4': row['y4'],
                'original_name': row['Image'],
                'source': 'dataset2'
            })
            image_counter += 1
        else:
            print(f"Warning: Image not found - {old_img_path}")
    
    # Create combined DataFrame and save to CSV
    combined_df = pd.DataFrame(combined_data)
    output_csv = os.path.join(output_dir, "combined_data.csv")
    combined_df.to_csv(output_csv, index=False)
    
    print(f"\nCombined dataset statistics:")
    print(f"Total images: {image_counter}")
    print(f"Images from dataset1: {len(combined_df[combined_df['source'] == 'dataset1'])}")
    print(f"Images from dataset2: {len(combined_df[combined_df['source'] == 'dataset2'])}")
    print(f"Skipped images from dataset1 (invalid coordinates): {skipped_count1}")
    print(f"Skipped images from dataset2 (invalid coordinates): {skipped_count2}")
    print(f"\nOutput saved to: {output_dir}")
    print(f"CSV file: {output_csv}")
    print(f"Images directory: {output_img_dir}")

if __name__ == "__main__":
    combine_datasets()