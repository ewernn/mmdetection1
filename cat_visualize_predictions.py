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
import argparse
from pycocotools.cocoeval import COCOeval

def create_model(device, num_classes):
    weights = ResNet152_Weights.DEFAULT
    backbone = resnet_fpn_backbone(backbone_name='resnet152', weights=weights, trainable_layers=3)
    
    anchor_sizes = (
        (160, 160),    # Very small objects
        (182, 182),  # Small objects
        (200, 200),  # Medium objects
        (222, 222),  # Large objects
        (251, 251),  # Very large objects
    )
    aspect_ratios = ((0.877, 0.8, 0.752, 0.704, 0.641),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    model = FasterRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator)
    
    for name, parameter in model.backbone.body.named_parameters():
        parameter.requires_grad = True
    
    return model

def modify_model(model, num_classes):
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Apply modifications with default values from train_cat_labelme.py
    model.rpn.nms_thresh = 0.9
    model.rpn.fg_iou_thresh = 0.85
    model.rpn.bg_iou_thresh = 0.1
    model.roi_heads.batch_size_per_image = 32
    model.roi_heads.positive_fraction = 0.3
    model.roi_heads.score_thresh = 0.45
    model.roi_heads.nms_thresh = 0.1
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

def calculate_mAP(coco_gt, coco_dt):
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats[0]  # mAP @ IoU=0.50:0.95

def main(model_path, coco_path, img_dir, save_dir, device):
    num_classes = 3
    model = load_model(model_path, device, num_classes)
    coco, img_ids = get_dataset(coco_path, img_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    coco_results = []
    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        path = img_info['file_name']
        image = Image.open(os.path.join(img_dir, path)).convert("L")  # Load as grayscale
        image_tensor = torch.from_numpy(np.array(image)).unsqueeze(0).float().div(255).unsqueeze(0).to(device)

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        gt_boxes = [ann['bbox'] for ann in anns]  # [x, y, width, height]
        
        with torch.no_grad():
            prediction = model(image_tensor)[0]

        pred_boxes = prediction['boxes'].cpu().numpy()
        pred_scores = prediction['scores'].cpu().numpy()
        pred_labels = prediction['labels'].cpu().numpy()
        
        # Filter predictions based on a score threshold
        score_threshold = 0.45  # Match the score_thresh from modify_model
        mask = pred_scores > score_threshold
        pred_boxes = pred_boxes[mask]
        pred_scores = pred_scores[mask]
        pred_labels = pred_labels[mask]

        visualize_and_save(image_tensor.squeeze(0), gt_boxes, pred_boxes, pred_scores, pred_labels, img_id, save_dir)
        
        for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
            x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min
            coco_results.append({
                'image_id': img_id,
                'category_id': int(label),
                'bbox': [x_min, y_min, width, height],
                'score': float(score)
            })

    coco_gt = COCO(coco_path)
    coco_dt = coco_gt.loadRes(coco_results)
    
    mAP = calculate_mAP(coco_gt, coco_dt)
    print(f"mAP @ IoU=0.50:0.95: {mAP:.4f}")

#/content/drive/MyDrive/MM/CatKidney/data/cat-dataset/ozt72mge/best_model.pth
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize predictions for cat kidney dataset")
    parser.add_argument('--runId', type=str, default='l5v1rtfk', help='Run ID for the model')

    args = parser.parse_args()

    #data_root = '/content/drive/MyDrive/MM/CatKidney/data/cat_kidney_dataset_csv_filtered/'
    #data_root = '/content/drive/MyDrive/MM/CatKidney/data/cat-data-combined-oct20/'
    data_root = '/content/drive/MyDrive/MM/CatKidney/data/a_fresh_cat_dataset/images/'
    #model_path = f"{data_root}{args.runId}/best_model.pth"
    model_path = f'/content/drive/MyDrive/MM/CatKidney/exps/{args.runId}/best_model.pth'
    #model_path = f'/content/drive/MyDrive/MM/CatKidney/data/cat_kidney_dataset_csv_filtered/3bwem77j/best_model.pth' # 3bwem77j from old sweep
    coco_path = data_root + 'test.json'
    img_dir = data_root
    save_dir = data_root + f'predictions_output-nov20-{args.runId}/'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(model_path, coco_path, img_dir, save_dir, device)
