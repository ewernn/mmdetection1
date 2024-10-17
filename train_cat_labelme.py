import os
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import wandb
import math
import argparse
import random
import torchvision
from torchvision.models import ResNet101_Weights, ResNet152_Weights, ResNet50_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms.functional import to_pil_image
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models.detection.rpn import AnchorGenerator
import ast
import torchvision.ops

## TTA USAGE ##
# --use_tta --tta_contrasts 0.8 1.0 1.2

# Initialize global variables
use_wandb = False
use_colab = True

class CocoDataset(Dataset):
    def __init__(self, root, annFile, transforms=None, preload=False, only_10=False, subfolder=''):
        self.root = os.path.join(root, subfolder)  # Include subfolder in the path
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        if only_10:
            random.shuffle(self.ids)
            self.ids = self.ids[:10]
        self.transforms = transforms
        self.preload = preload
        self.images = {}
        if self.preload:
            self._preload_images()
    def _preload_images(self):
        for img_id in self.ids:
            img_info = self.coco.loadImgs(img_id)[0]
            path = img_info['file_name']
            img = Image.open(os.path.join(self.root, path)).convert("L")  # Correct path with subfolder
            self.images[img_id] = img

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        img_info = coco.loadImgs(img_id)[0]
        path = img_info['file_name']
        if self.preload:
            img = self.images[img_id]
        else:
            img = Image.open(os.path.join(self.root, path)).convert("L")  # Correct path with subfolder
        num_objs = len(anns)
        boxes = []
        labels = []
        for i in range(num_objs):
            xmin = anns[i]['bbox'][0]
            ymin = anns[i]['bbox'][1]
            xmax = xmin + anns[i]['bbox'][2]
            ymax = ymin + anns[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])  # go from (x1,y1,w,h) to (x1,y1,x2,y2)
            labels.append(anns[i]['category_id'])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        target = {}
        target["boxes"] = boxes
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)  # Ensure labels is a tensor
        target["image_id"] = torch.tensor([img_id])
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        # Ensure all items in target are tensors
        target = {k: torch.as_tensor(v) for k, v in target.items()}
        return img, target
    
    def __len__(self):
        return len(self.ids)

def adjust_brightness(img):
    return TF.adjust_brightness(img, brightness_factor=random.uniform(0.3, 1.5))

def expand_channels(img):
    return img.repeat(3, 1, 1) if img.shape[0] == 1 else img

def adjust_contrast(img):
    return TF.adjust_contrast(img, contrast_factor=random.uniform(0.3, 1.5))

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class RandomAffine:
    def __init__(self, degrees, translate=None, scale=None, fill=0):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.fill = fill

    def __call__(self, image, target):
        _, height, width = image.shape

        angle = random.uniform(self.degrees[0], self.degrees[1])

        if self.translate is not None:
            max_dx = self.translate[0] * width
            max_dy = self.translate[1] * height
            translations = (random.uniform(-max_dx, max_dx),
                            random.uniform(-max_dy, max_dy))
        else:
            translations = (0, 0)

        scale = random.uniform(self.scale[0], self.scale[1]) if self.scale is not None else 1.0

        matrix = self._get_affine_matrix(center=(width / 2, height / 2), angle=angle, translations=translations, scale=scale)

        image = TF.affine(image, angle, translations, scale, 0, fill=self.fill)

        if "boxes" in target:
            boxes = target["boxes"]
            transformed_boxes = []
            for box in boxes:
                transformed_box = self._transform_bbox(box, matrix, width, height)
                transformed_boxes.append(transformed_box)
            target["boxes"] = torch.stack(transformed_boxes)

        return image, target

    def _get_affine_matrix(self, center, angle, translations, scale):
        angle = math.radians(angle)

        rot_mat = torch.tensor([
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)]
        ])

        rot_mat = scale * rot_mat

        matrix = torch.eye(3)
        matrix[:2, :2] = rot_mat
        matrix[0, 2] = translations[0]
        matrix[1, 2] = translations[1]

        matrix = torch.tensor([
            [1, 0, center[0]],
            [0, 1, center[1]],
            [0, 0, 1]
        ]) @ matrix @ torch.tensor([
            [1, 0, -center[0]],
            [0, 1, -center[1]],
            [0, 0, 1]
        ])

        return matrix

    def _transform_bbox(self, bbox, matrix, width, height):
        x1, y1, x2, y2 = bbox
        points = torch.tensor([
            [x1, y1, 1],
            [x2, y1, 1],
            [x2, y2, 1],
            [x1, y2, 1],
        ])

        transformed_points = torch.matmul(points, matrix.t())

        new_x1 = transformed_points[:, 0].min()
        new_y1 = transformed_points[:, 1].min()
        new_x2 = transformed_points[:, 0].max()
        new_y2 = transformed_points[:, 1].max()

        new_x1 = torch.clamp(new_x1, 0, width)
        new_y1 = torch.clamp(new_y1, 0, height)
        new_x2 = torch.clamp(new_x2, 0, width)
        new_y2 = torch.clamp(new_y2, 0, height)

        new_x2 = torch.max(new_x2, new_x1 + 1)
        new_y2 = torch.max(new_y2, new_y1 + 1)

        return torch.tensor([new_x1, new_y1, new_x2, new_y2])

class AdjustBrightness:
    def __init__(self, brightness_range):
        self.brightness_range = brightness_range

    def __call__(self, image, target):
        brightness_factor = random.uniform(self.brightness_range[0], self.brightness_range[1])
        image = TF.adjust_brightness(image, brightness_factor)
        return image, target

class AdjustContrast:
    def __init__(self, contrast_range):
        self.contrast_range = contrast_range

    def __call__(self, image, target):
        contrast_factor = random.uniform(self.contrast_range[0], self.contrast_range[1])
        image = TF.adjust_contrast(image, contrast_factor)
        return image, target

class GaussianBlur:
    def __init__(self, kernel_size, sigma):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, image, target):
        image = TF.gaussian_blur(image, self.kernel_size, self.sigma)
        return image, target

class RandomAdjustSharpness:
    def __init__(self, sharpness_factor, p=0.5):
        self.sharpness_factor = sharpness_factor
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            image = TF.adjust_sharpness(image, self.sharpness_factor)
        return image, target

class CustomToTensor:
    def __call__(self, image, target):
        image = TF.to_tensor(image)
        return image, target

def get_transform(train, brightness_range, contrast_range):
    transforms = []
    transforms.append(CustomToTensor())
    if train:
        transforms.extend([
            RandomAffine(degrees=(-5, 5), translate=(0.1, 0.1), scale=(0.9, 1.1), fill=0),
            AdjustBrightness(brightness_range),
            AdjustContrast(contrast_range),
            GaussianBlur(kernel_size=random.choice([3, 5, 7]), sigma=(0.1, 2.5)),
            RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        ])
    transforms.append(lambda img, target: (expand_channels(img), target))
    return Compose(transforms)

def collate_fn(batch):
    return tuple(zip(*batch))

def visualize_boxes(image, gt_boxes, gt_labels, pred_boxes, pred_labels, image_id, save_path):
    """
    Visualize ground truth and predicted bounding boxes on the image and save it.
    """

    if pred_boxes.size(0) == 0:
        print("No predicted boxes to visualize.")
        return

    image_pil = to_pil_image(image.cpu())
    
    gt_boxes = gt_boxes.cpu()
    gt_labels = gt_labels.cpu()
    pred_boxes = pred_boxes.cpu()
    pred_labels = pred_labels.cpu()

    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image_pil)
    
    for box, label in zip(gt_boxes, gt_labels):
        x, y, w, h = box[0].item(), box[1].item(), (box[2]-box[0]).item(), (box[3]-box[1]).item()
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y, f'GT: {label.item()}', color='g', fontsize=10, verticalalignment='top')
    
    if len(pred_boxes) > 0:
        for box, label in zip(pred_boxes, pred_labels):
            x, y, w, h = box[0].item(), box[1].item(), (box[2]-box[0]).item(), (box[3]-box[1]).item()
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x, y, f'Pred: {label.item()}', color='r', fontsize=10, verticalalignment='top')
    else:
        ax.text(10, 10, 'No predictions', color='r', fontsize=12, verticalalignment='top')
    
    ax.set_xticks([])
    ax.set_yticks([])
    plt.title(f'Image ID: {image_id}')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

def filter_kidney_predictions(boxes, scores, labels, iou_threshold=0.5):
    if boxes.shape[0] == 0:
        return boxes, scores, labels

    best_indices = {}

    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        label = label.item()
        if label not in best_indices or score > scores[best_indices[label]]:
            best_indices[label] = i

    best_indices = list(best_indices.values())
    best_boxes = boxes[best_indices]
    best_scores = scores[best_indices]
    best_labels = labels[best_indices]

    return best_boxes, best_scores, best_labels

def tta_inference(model, image, device, contrast_factors, brightness_factors):
    original_image = image.clone()
    predictions = []
    
    for contrast_factor in contrast_factors:
        for brightness_factor in brightness_factors:
            augmented_image = TF.adjust_contrast(original_image, contrast_factor=contrast_factor)
            augmented_image = TF.adjust_brightness(augmented_image, brightness_factor=brightness_factor)
            
            augmented_image = augmented_image.to(device)
            
            with torch.no_grad():
                prediction = model([augmented_image])[0]
            
            predictions.append(prediction)
    
    return aggregate_predictions(predictions)

def aggregate_predictions(predictions):
    all_boxes = torch.cat([p['boxes'] for p in predictions])
    all_scores = torch.cat([p['scores'] for p in predictions])
    all_labels = torch.cat([p['labels'] for p in predictions])
    
    keep = torchvision.ops.batched_nms(all_boxes, all_scores, all_labels, iou_threshold=0.5)
    
    return {
        'boxes': all_boxes[keep],
        'scores': all_scores[keep],
        'labels': all_labels[keep]
    }

def evaluate(model, data_loader, device, epoch, args):
    model.eval()
    coco = data_loader.dataset.coco
    coco_results = []
    if use_colab:
        save_dir = f'/content/drive/MyDrive/MM/CatKidney/exps/imgs_out/epoch_{epoch}'
    else:
        save_dir = f'/Users/ewern/Desktop/code/MetronMind/exps/cat_kidneys/images_with_predicted_bboxes/epoch_{epoch}'
    os.makedirs(save_dir, exist_ok=True)
    
    for i, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        
        for j, (image, target) in enumerate(zip(images, targets)):
            if args.use_tta:
                output = tta_inference(model, image, device, contrast_factors=args.tta_contrasts, brightness_factors=args.tta_brightness)
            else:
                with torch.no_grad():
                    output = model([image])[0]
            
            image_id = target["image_id"].item()
            boxes = output["boxes"]
            scores = output["scores"]
            labels = output["labels"]
            
            boxes, scores, labels = filter_kidney_predictions(boxes, scores, labels)
            
            # if i * len(images) + j < 10:
            #     print(f"Image {i * len(images) + j + 1}: boxes: {boxes}, scores: {scores}, labels: {labels}")
            
            if len(boxes) > 0:
                for box, score, label in zip(boxes, scores, labels):
                    coco_results.append({
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])],
                        "score": float(score),
                    })
    
    if len(coco_results) == 0:
        print("No valid detections found. Returning 0 mAP.")
        return {"mAP": 0.0, "AP_50": 0.0, "AP_75": 0.0}
    
    coco_dt = coco.loadRes(coco_results)
    coco_eval = COCOeval(coco, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    metrics = {
        "mAP": coco_eval.stats[0],
        "AP_50": coco_eval.stats[1],
        "AP_75": coco_eval.stats[2],
        "AP_small": coco_eval.stats[3],
        "AP_medium": coco_eval.stats[4],
        "AP_large": coco_eval.stats[5],
        "AR_max_1": coco_eval.stats[6],
        "AR_max_10": coco_eval.stats[7],
        "AR_max_100": coco_eval.stats[8],
        "AR_small": coco_eval.stats[9],
        "AR_medium": coco_eval.stats[10],
        "AR_large": coco_eval.stats[11],
    }

    print("\nDetailed Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    if use_wandb:
        wandb.log(metrics)

    return metrics

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_idx, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch + 1}: Avg Loss = {avg_loss:.4f}, Learning Rate = {current_lr:.6f}")

    return avg_loss

def parse_tuple(argument):
    try:
        return ast.literal_eval(argument)
    except ValueError as e:
        raise argparse.ArgumentTypeError("Invalid tuple: %s" % (e,))

def create_model(args, num_classes):
    weights = ResNet50_Weights.DEFAULT
    if args.backbone == 'resnet101': weights = ResNet101_Weights.DEFAULT
    if args.backbone == 'resnet152': weights = ResNet152_Weights.DEFAULT

    if args.gradual_unfreeze:
        trainable_layers = 0  # Start with all layers frozen
    else:
        trainable_layers = 5  # Make all layers trainable

    backbone = resnet_fpn_backbone(backbone_name=args.backbone, weights=weights, trainable_layers=trainable_layers)

    anchor_sizes = (
        (89, 89),    # Very small objects
        (112, 112),  # Small objects
        (141, 141),  # Medium objects
        (178, 178),  # Large objects
        (225, 225),  # Very large objects
    )
    aspect_ratios = ((0.69, 0.74, 0.79, 0.84, 1.0),) * len(anchor_sizes)

    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    model = FasterRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator)

    # Ensure all parameters are trainable if not using gradual unfreeze
    if not args.gradual_unfreeze:
        for param in model.parameters():
            param.requires_grad = True

    return model

def modify_model(model, num_classes, args):
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Increase NMS threshold to be more selective
    model.rpn.nms_thresh = args.rpn_nms_thresh  # Original: 0.7

    # Increase IoU thresholds for foreground and background
    model.rpn.fg_iou_thresh = args.rpn_fg_iou_thresh  # Original: 0.7
    model.rpn.bg_iou_thresh = args.rpn_bg_iou_thresh  # Original: 0.3

    # Reduce batch size and positive fraction to be more selective
    model.roi_heads.batch_size_per_image = args.roi_heads_batch_size_per_image  # Original: 8
    model.roi_heads.positive_fraction = args.roi_heads_positive_fraction  # Original: 0.25

    # Increase score threshold for higher confidence
    model.roi_heads.score_thresh = args.roi_heads_score_thresh  # Original: 0.6

    # Slightly increase NMS threshold for ROI heads
    model.roi_heads.nms_thresh = args.roi_heads_nms_thresh  # Original: 0.4

    # Increase detections per image
    model.roi_heads.detections_per_img = args.roi_heads_detections_per_img  # Original: 1

    # Adjust pre and post NMS top N
    model.rpn.pre_nms_top_n = lambda: args.rpn_pre_nms_top_n  # Original: 150
    model.rpn.post_nms_top_n = lambda: args.rpn_post_nms_top_n  # Original: 20

    return model

def load_checkpoint(filepath, model, optimizer):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    best_mAP = checkpoint.get('best_mAP', 0.0)
    return model, optimizer, epoch, best_mAP

def setup_environment(args):
    if args.colab:
        data_root = '/content/drive/MyDrive/MM/CatKidney/data/cat-data-combined-oct9/'
        chkpt_dir = '/content/drive/MyDrive/MM/CatKidney/exps/'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        c2 = '/Users/ewern/Desktop/code/MetronMind/c2/'
        data_root = c2
        chkpt_dir = c2
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    return data_root, chkpt_dir, device

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train C2 Detection Model')
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--colab', action='store_true', help='Use Google Colab data path')
    parser.add_argument('--only_10', action='store_true', help='Use only 10 samples for quick testing')
    parser.add_argument('--backbone', type=str, default='resnet152', choices=['resnet50', 'resnet101', 'resnet152'],
                        help='Backbone architecture to use')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for training')
    parser.add_argument('--no_sweep', action='store_true', help='Disable wandb sweep and use specified hyperparameters')
    parser.add_argument('--no_preload', action='store_true', help='Preload images into memory')
    parser.add_argument('--all_images', action='store_true', help='use all images in dataloaders (including NaN entries)')
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume from')
    parser.add_argument('--rpn_nms_thresh', type=float, default=0.3, help='RPN NMS threshold')
    parser.add_argument('--roi_heads_nms_thresh', type=float, default=0.1, help='ROI heads NMS threshold')
    parser.add_argument('--roi_heads_score_thresh', type=float, default=0.45, help='ROI heads score threshold')
    parser.add_argument('--rpn_bg_iou_thresh', type=float, default=0.1, help='RPN background IoU threshold')
    parser.add_argument('--rpn_fg_iou_thresh', type=float, default=0.85, help='RPN foreground IoU threshold')
    parser.add_argument('--roi_heads_batch_size_per_image', type=int, default=32, help='ROI heads batch size per image')
    parser.add_argument('--roi_heads_positive_fraction', type=float, default=0.3, help='Fraction of positive ROIs')
    parser.add_argument('--roi_heads_detections_per_img', type=int, default=4, help='Number of detections per image')
    parser.add_argument('--rpn_pre_nms_top_n', type=int, default=200, help='RPN pre-NMS top N')
    parser.add_argument('--rpn_post_nms_top_n', type=int, default=50, help='RPN post-NMS top N')
    parser.add_argument('--score_thresh', type=float, default=0.68, help='Score threshold for detections')
    parser.add_argument('--num_epochs', type=int, default=101, help='Number of epochs to train for')
    parser.add_argument('--brightness_min', type=float, default=0.38, help='Minimum brightness factor')
    parser.add_argument('--brightness_max', type=float, default=1.42, help='Maximum brightness factor')
    parser.add_argument('--contrast_min', type=float, default=0.44, help='Minimum contrast factor')
    parser.add_argument('--contrast_max', type=float, default=1.62, help='Maximum contrast factor')
    parser.add_argument('--use_tta', action='store_true', help='Use Test Time Augmentation during evaluation')
    parser.add_argument('--tta_contrasts', nargs='+', type=float, default=[0.5, 1.0, 1.5], help='Contrast factors for TTA (space-separated list of floats)')
    parser.add_argument('--tta_brightness', nargs='+', type=float, default=[0.5, 1.0, 1.5], help='Brightness factors for TTA (space-separated list of floats)')
    parser.add_argument('--gradual_unfreeze', action='store_true', help='Use gradual unfreezing strategy')
    parser.add_argument('--unfreeze_schedule', type=str, default='10:1,20:2,30:3,40:4', help='Schedule for unfreezing layers (epoch:num_layers,...)')
    parser.add_argument('--max_unfrozen_layers', type=int, default=4, help='Maximum number of layers to unfreeze')
    parser.add_argument('--lr_decrease_ratio', type=float, default=100, help='Ratio for learning rate decrease (e.g., 50 to 1000)')
    parser.add_argument('--unfreeze_start_epoch', type=int, default=20, help='Epoch to start unfreezing layers')
    parser.add_argument('--unfreeze_frequency', type=int, default=20, help='Frequency of unfreezing layers (in epochs)')
    parser.add_argument('--unfreeze_lr_multiplier', type=float, default=0.05, help='Learning rate multiplier for unfrozen layers')
    return parser.parse_args()


def unfreeze_layers(model, num_layers):
    for name, param in model.backbone.body.named_parameters():
        param.requires_grad = False
    
    layers_to_unfreeze = list(model.backbone.body.named_children())[-num_layers:]
    for layer in layers_to_unfreeze:
        for param in layer[1].parameters():
            param.requires_grad = True

    return model

def main():
    global use_wandb, use_colab

    args = parse_arguments()

    eval_every_n_epochs = 1
    num_classes = 3
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    min_lr = args.learning_rate / args.lr_decrease_ratio
    data_root, checkpoint_dir, device = setup_environment(args)

    use_wandb = args.wandb
    use_colab = args.colab
    only_10 = args.only_10

    if use_wandb:
        wandb.init(project="kidney_detection", config=args)
    if only_10:
        num_epochs = min(num_epochs, 10)

    print("Initializing datasets...")
    if args.all_images:
        train_ann_file = data_root + 'coco_output/two_only_train_Data_coco_format.json'
        val_ann_file = data_root + 'coco_output/two_only_val_Data_coco_format.json'
    else:
        train_ann_file = data_root + 'coco_output/two_only_train_Data_coco_format.json'
        val_ann_file = data_root + 'coco_output/two_only_val_Data_coco_format.json'

    preload = not args.no_preload
    brightness_range = (args.brightness_min, args.brightness_max)
    contrast_range = (args.contrast_min, args.contrast_max)
    train_dataset = CocoDataset(data_root, train_ann_file, transforms=get_transform(train=True, brightness_range=brightness_range, contrast_range=contrast_range), preload=preload, only_10=only_10)#, subfolder='Train')
    val_dataset = CocoDataset(data_root, val_ann_file, transforms=get_transform(train=False, brightness_range=brightness_range, contrast_range=contrast_range), preload=preload, only_10=only_10)#, subfolder='Test')

    print("Creating data loaders...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, collate_fn=collate_fn)
    print("Data loaders created.")

    print("Creating model...")
    model = create_model(args, num_classes)

    print("Modifying model parameters...")
    model = modify_model(model, num_classes, args)

    print("Printing trainable status of layers:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.requires_grad}")

    print(f"Using device: {device}")

    print("Moving model to device...")
    model.to(device)
    print("Model moved to device.")

    print("Creating optimizer and scheduler...")
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0001)

    start_epoch = 0
    best_mAP = 0.0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from {args.resume}")
            model, optimizer, start_epoch, best_mAP = load_checkpoint(args.resume, model, optimizer)
            print(f"Resuming training from epoch {start_epoch} with best mAP {best_mAP}")
        else:
            print(f"No checkpoint found at {args.resume}")

    lr_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - start_epoch, eta_min=min_lr)

    print("Optimizer and scheduler created.")

    if use_wandb:
        run_id = wandb.run.id
        checkpoint_dir = os.path.join(checkpoint_dir, run_id)
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_epoch = -1

    print(f"Validation dataset size: {len(val_dataset)}")

    if args.gradual_unfreeze:
        unfreeze_schedule = {
            10: 1,
            20: 2,
            30: 3,
            40: 4,
        }
    else:
        unfreeze_schedule = {}

    for epoch in range(start_epoch, num_epochs):
        if args.gradual_unfreeze and epoch in unfreeze_schedule:
            model = unfreeze_layers(model, unfreeze_schedule[epoch])
            # Recreate optimizer with different learning rates
            params = [
                {'params': [p for n, p in model.named_parameters() if 'backbone' not in n], 'lr': learning_rate},
                {'params': [p for n, p in model.named_parameters() if 'backbone' in n], 'lr': learning_rate * 0.1},
            ]
            optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0001)
            lr_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - epoch, eta_min=min_lr)

        if epoch < start_epoch + 5:
            lr = learning_rate * ((epoch - start_epoch + 1) / 5)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            lr_scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}, Current learning rate: {current_lr}")

        avg_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=50)

        if (epoch + 1) % eval_every_n_epochs == 0:
            metrics = evaluate(model, val_loader, device, epoch, args)
            mAP = metrics.get("mAP", 0.0)

            print(f"Epoch {epoch + 1}: mAP = {mAP:.4f}, Avg Loss = {avg_loss:.4f}")

            if mAP == 0.0:
                print("Warning: Model failed to detect any objects. Consider adjusting model parameters or checking the dataset.")

            if use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "training_loss": avg_loss,
                    "learning_rate": current_lr,
                    "mAP": mAP,
                    "AP_50": metrics["AP_50"],
                    "AP_75": metrics["AP_75"],
                })

            if mAP > best_mAP:
                best_mAP = mAP
                best_epoch = epoch + 1
                print(f"New best mAP: {best_mAP:.4f}")

                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_mAP': best_mAP,
                }, os.path.join(checkpoint_dir, 'best_model.pth'))

                print(f"New best model saved in: {checkpoint_dir}/best_model.pth")
            else:
                print(f"mAP did not improve. Best is still {best_mAP:.4f} from epoch {best_epoch}")

    print(f"Training complete. Best mAP: {best_mAP:.4f} at epoch {best_epoch}")

    if use_wandb:
        wandb.log({"best_mAP": best_mAP, "best_epoch": best_epoch})
        wandb.finish()

if __name__ == "__main__":
    main()