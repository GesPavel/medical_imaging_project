import json
import numpy as np
import SimpleITK as sitk
from samed2.predict import MedicalSegmenter
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from consts import MR_PATH, PART3_ARTIFACTS_DIR

def _resize_mask_nearest(mask, target_shape):
    if mask.shape == target_shape:
        return mask
    mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
    resized = mask_img.resize((target_shape[1], target_shape[0]), resample=Image.NEAREST)
    return np.array(resized) > 0

def _bbox_mask(bbox, shape):
    h, w = shape
    x_min = int(np.floor(bbox["x_min"]))
    y_min = int(np.floor(bbox["y_min"]))
    x_max = int(np.ceil(bbox["x_max"]))
    y_max = int(np.ceil(bbox["y_max"]))
    x_min = max(0, min(w, x_min))
    x_max = max(0, min(w, x_max))
    y_min = max(0, min(h, y_min))
    y_max = max(0, min(h, y_max))
    mask = np.zeros((h, w), dtype=bool)
    if x_max > x_min and y_max > y_min:
        mask[y_min:y_max, x_min:x_max] = True
    return mask

def load_mr_slice(z_index):
    mr_img = sitk.ReadImage(str(MR_PATH))
    mr_array = sitk.GetArrayFromImage(mr_img)
    # The arrays are in (Z, Y, X)
    if z_index >= mr_array.shape[0]:
        z_index = mr_array.shape[0] // 2
    return mr_array[z_index]
    
def normalize_for_sam(image_slice):
    p1, p99 = np.percentile(image_slice, (1, 99))
    norm = np.clip(image_slice, p1, p99)
    norm = (norm - p1) / (p99 - p1 + 1e-8)
    
    # SAM expects 3-channel uint8 [0, 255]
    norm_8bit = (norm * 255).astype(np.uint8)
    rgb_img = np.stack([norm_8bit]*3, axis=-1)
    return rgb_img

def main():
    json_path = PART3_ARTIFACTS_DIR / "pet_registered_bbox.json"
    if not json_path.exists():
        print(f"Error: {json_path} not found. Run part3.py first.")
        return

    with open(json_path, "r") as f:
        bbox_data = json.load(f)

    # 1. Load the corresponding MR slice based on the JSON
    z_index = bbox_data.get("slice_index", 0)
    mr_slice = load_mr_slice(z_index)
    
    # Convert bbox to SAM format: [x_min, y_min, x_max, y_max]
    b = bbox_data["bbox"]
    h, w = mr_slice.shape

    sam_size = 1024

    scale_x = sam_size / w
    scale_y = sam_size / h

    input_box = np.array([
        b["x_min"] * scale_x,
        b["y_min"] * scale_y,
        b["x_max"] * scale_x,
        b["y_max"] * scale_y,
    ], dtype=np.float32)
    # 2. Prepare imagery for SAMed-2
    image_rgb = normalize_for_sam(mr_slice)
    image_pil = Image.fromarray(image_rgb)

    # 3. Model setup for SAMed-2
    # Be sure to point to your actual SAMed-2 checkpoint file

    segmenter = MedicalSegmenter(
        model_type='samed2',
        checkpoint_path='latest_epoch_0217.pth'
    )

    # Segment
    result = segmenter.predict(image_pil,
        box=input_box
    )

    iou_predict = result['iou']

    # Visualize
    segmenter.visualize(
        image_pil,
        result['mask'],
        PART3_ARTIFACTS_DIR / 'result.jpg'
    )

    # Additional visualization: bbox on top of segmentation
    seg_mask = result["mask"]
    if seg_mask.ndim == 3:
        seg_mask = seg_mask[0]
    seg_mask = _resize_mask_nearest(seg_mask, mr_slice.shape)

    # Load user-provided ground-truth mask and compute IoU vs segmentation
    gt_path = PART3_ARTIFACTS_DIR / "mr_slice_gt_mask.npy"
    if not gt_path.exists():
        print(f"Error: {gt_path} not found. Run ground_truth_generation.py first.")
        return

    gt_mask = np.load(gt_path).astype(bool)
    if gt_mask.shape != mr_slice.shape:
        gt_mask = _resize_mask_nearest(gt_mask, mr_slice.shape)

    bbox_mask = _bbox_mask(b, mr_slice.shape)
    seg_in_bbox = seg_mask[bbox_mask]
    gt_in_bbox = gt_mask[bbox_mask]

    tp = int(np.logical_and(seg_in_bbox, gt_in_bbox).sum())
    fp = int(np.logical_and(seg_in_bbox, ~gt_in_bbox).sum())
    fn = int(np.logical_and(~seg_in_bbox, gt_in_bbox).sum())
    tn = int(np.logical_and(~seg_in_bbox, ~gt_in_bbox).sum())
    total = tp + fp + fn + tn

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(mr_slice, cmap="gray")
    ax.imshow(gt_mask, cmap="Blues", alpha=0.35)
    ax.imshow(seg_mask, cmap="magma", alpha=0.35)
    rect = Rectangle(
        (b["x_min"], b["y_min"]),
        b["x_max"] - b["x_min"],
        b["y_max"] - b["y_min"],
        linewidth=2,
        edgecolor="lime",
        facecolor="none",
    )
    ax.add_patch(rect)
    ax.set_title("MR + GT mask + SAMed-2 mask + bbox")
    ax.axis("off")

    PART3_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    overlay_path = PART3_ARTIFACTS_DIR / "bbox_on_mask.jpg"
    fig.savefig(overlay_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    model_iou = float(np.atleast_1d(iou_predict)[0])
    print(f"Saved bbox+mask overlay to {overlay_path}")
    print(f"Within-bbox TP/FP/FN/TN: {(tp/total):.4f}/{(fp/total):.4f}/{(fn/total):.4f}/{(tn/total):.4f}")
    print(f"Within-bbox precision: {precision:.4f}")
    print(f"Within-bbox recall: {recall:.4f}")
    print(f"Within-bbox accuracy: {accuracy:.4f}")
    print(f"Within-bbox F1: {f1:.4f}")
    print(f"Within-bbox IoU: {iou:.4f}")
    print(f"Model-reported IoU: {model_iou:.4f}")

if __name__ == "__main__":
    main()
