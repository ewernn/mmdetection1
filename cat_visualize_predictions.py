import numpy as np
import torch
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
from pycocotools.coco import COCO
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models import ResNet152_Weights

def create_model(device, num_classes):
    weights = ResNet152_Weights.DEFAULT
    backbone = resnet_fpn_backbone(backbone_name='resnet152', weights=weights, trainable_layers=3)
    
    anchor_sizes = (
        (89, 89),    # Very small objects
        (112, 112),  # Small objects
        (141, 141),  # Medium objects
        (178, 178),  # Large objects
        (225, 225),  # Very large objects
    )
    aspect_ratios = ((0.71, 0.75, 0.79, 0.84, 1.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    model = FasterRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator)
    
    for name, parameter in model.backbone.body.named_parameters():
        parameter.requires_grad = True
    
    return model

def modify_model(model, num_classes):
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Apply modifications with hardcoded values (matching the defaults in train_cat.py)
    model.rpn.nms_thresh = 0.3#0.6
    model.rpn.fg_iou_thresh = 0.8
    model.rpn.bg_iou_thresh = 0.5
    model.roi_heads.batch_size_per_image = 32
    model.roi_heads.positive_fraction = 0.3
    model.roi_heads.score_thresh = 0.6
    model.roi_heads.nms_thresh = 0.1#0.3
    model.roi_heads.detections_per_img = 4

    model.rpn.pre_nms_top_n = lambda: 200
    model.rpn.post_nms_top_n = lambda: 50

    return model

def load_model(model_path, device, num_classes):
    model = create_model(device, num_classes)
    model = modify_model(model, num_classes)
    model.to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def get_dataset(coco_path, img_dir):
    coco = COCO(coco_path)
    img_ids = coco.getImgIds()
    return coco, img_ids

def visualize_and_save(image, gt_boxes, pred_boxes, pred_scores, pred_labels, image_id, save_dir):
    # Convert tensor image to PIL Image
    image_pil = to_pil_image(image.cpu())

    # Create figure and axis
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image_pil, cmap='gray')

    # Draw ground truth boxes in green
    for box in gt_boxes:
        x_min, y_min, width, height = box
        x_max = x_min + width
        y_max = y_min + height
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(rect)

    # Draw predicted boxes in red and display scores and labels
    for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x_max, y_max, f'Class {label}: {score:.3f}', color='red', fontsize=10, verticalalignment='top', backgroundcolor='white')

    ax.set_xticks([])
    ax.set_yticks([])
    plt.title(f'Image ID: {image_id}')
    plt.savefig(os.path.join(save_dir, f'image_{image_id}.png'), bbox_inches='tight')
    plt.close(fig)

def main(model_path, coco_path, img_dir, save_dir, device):
    num_classes = 3
    model = load_model(model_path, device, num_classes)
    coco, img_ids = get_dataset(coco_path, img_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        path = img_info['file_name']
        image = Image.open(os.path.join(img_dir, path)).convert("L")  # Convert image to grayscale
        image = image.convert("RGB")  # Convert grayscale image to RGB by replicating channels
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        gt_boxes = [ann['bbox'] for ann in anns]  # [x, y, width, height]
        model.to(device)
        with torch.no_grad():
            prediction = model(image_tensor)[0]

        pred_boxes = prediction['boxes'].cpu().numpy()
        pred_scores = prediction['scores'].cpu().numpy()
        pred_labels = prediction['labels'].cpu().numpy()
        
        # Filter predictions based on a score threshold
        score_threshold = 0.5
        mask = pred_scores > score_threshold
        pred_boxes = pred_boxes[mask]
        pred_scores = pred_scores[mask]
        pred_labels = pred_labels[mask]

        visualize_and_save(image_tensor.squeeze(0), gt_boxes, pred_boxes, pred_scores, pred_labels, img_id, save_dir)

#/content/drive/MyDrive/MM/CatKidney/data/cat-dataset/ozt72mge/best_model.pth
if __name__ == "__main__":
    data_root = '/content/drive/MyDrive/MM/CatKidney/data/cat_kidney_dataset_csv_filtered/'
    model_path = data_root + 'l5v1rtfk/best_model.pth'
    coco_path = data_root + 'COCO_2/val_Data_coco_format-labelme.json'
    img_dir = data_root
    save_dir = data_root + 'predictions_output-sep20/'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(model_path, coco_path, img_dir, save_dir, device)
    