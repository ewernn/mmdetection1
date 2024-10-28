import os
import random
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import importlib
import train_cat_labelme
importlib.reload(train_cat_labelme)
from train_cat_labelme import CocoDataset, get_transform

def visualize_and_save_images(dataset, save_dir, num_images=5, with_augmentation=True):
    os.makedirs(save_dir, exist_ok=True)
    
    successful_saves = 0
    attempts = 0
    max_attempts = num_images * 3  # Limit total attempts to avoid infinite loop
    
    while successful_saves < num_images and attempts < max_attempts:
        try:
            idx = random.randint(0, len(dataset) - 1)
            
            # Get original filename regardless of augmentation
            img_info = dataset.coco.loadImgs(dataset.ids[idx])[0]
            original_filename = img_info['file_name']
            
            if with_augmentation:
                img, target = dataset[idx]
                print(f"Augmented image {successful_saves+1} ({original_filename}):")
                print(f"Image shape: {img.shape}")
                print(f"Target keys: {target.keys()}")
                print(f"Boxes: {target['boxes']}")
                print(f"Labels: {target['labels']}")
                img = to_pil_image(img)
            else:
                img_path = os.path.join(dataset.root, original_filename)
                
                # Check if image exists before trying to open it
                if not os.path.exists(img_path):
                    print(f"Skipping missing image: {img_path}")
                    attempts += 1
                    continue
                    
                img = Image.open(img_path).convert("RGB")
                target = dataset.coco.loadAnns(dataset.coco.getAnnIds(imgIds=dataset.ids[idx]))
                print(f"Original image {successful_saves+1} ({original_filename}):")
                print(f"Image size: {img.size}")
                print(f"Number of annotations: {len(target)}")
            
            fig, ax = plt.subplots(1, figsize=(12, 8))
            ax.imshow(img)
            
            # Add filename text at the bottom of the image
            ax.text(0.5, 0.02, original_filename, 
                   horizontalalignment='center',
                   verticalalignment='bottom',
                   transform=ax.transAxes,
                   color='white',
                   bbox=dict(facecolor='black', alpha=0.7))
            
            if with_augmentation:
                for box in target['boxes']:
                    x, y, w, h = box[0].item(), box[1].item(), (box[2] - box[0]).item(), (box[3] - box[1]).item()
                    if w > 1 and h > 1:  # Only draw boxes larger than 1x1 pixel
                        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
                        ax.add_patch(rect)
                        print(f"Drawing box: {x}, {y}, {w}, {h}")
                    else:
                        print(f"Skipping small box: {x}, {y}, {w}, {h}")
            else:
                for ann in target:
                    x, y, w, h = ann['bbox']
                    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
                    print(f"Drawing box: {x}, {y}, {w}, {h}")
            
            ax.set_title(f"{'Augmented' if with_augmentation else 'Original'} Image {original_filename}")
            plt.axis('off')
            
            # Save with original filename included in saved filename
            base_filename = os.path.splitext(original_filename)[0]
            save_path = os.path.join(save_dir, 
                                   f"{'aug' if with_augmentation else 'orig'}_{base_filename}_{successful_saves+1}.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
            
            successful_saves += 1
            
        except Exception as e:
            print(f"Error processing image at index {idx}: {str(e)}")
        
        attempts += 1
    
    print(f"Successfully saved {successful_saves} {'augmented' if with_augmentation else 'original'} images to {save_dir}")
    if successful_saves < num_images:
        print(f"Warning: Only able to save {successful_saves}/{num_images} images after {attempts} attempts")

# Paths from combine_datasets_and_convert_to_COCO.py
OUTPUT_PATH = '/Users/ewern/Desktop/code/MetronMind/data/cat-data-combined-oct20'
data_root = OUTPUT_PATH
train_ann_file = os.path.join(OUTPUT_PATH, 'val.json')
preload = False
only_10 = False

# Create dataset with specified brightness and contrast ranges
brightness_range = (0.36, 1.1)
contrast_range = (0.69, 1.58)
train_dataset = CocoDataset(data_root, train_ann_file, transforms=get_transform(train=True, 
                          brightness_range=brightness_range, contrast_range=contrast_range), 
                          preload=preload, only_10=only_10)

# Visualize and save augmented images
save_dir_aug = '/Users/ewern/Desktop/code/MetronMind/exps/coco-oct27-visualized-aug'
visualize_and_save_images(train_dataset, save_dir_aug, num_images=100, with_augmentation=True)

# Visualize and save original images (without augmentation)
save_dir_no_aug = '/Users/ewern/Desktop/code/MetronMind/exps/coco-oct27-visualized-no-aug'
visualize_and_save_images(train_dataset, save_dir_no_aug, num_images=100, with_augmentation=False)