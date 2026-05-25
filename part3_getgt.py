import json

import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path as MplPath
import numpy as np
import SimpleITK as sitk

from consts import PART3_ARTIFACTS_DIR, MR_PATH


def _draw_freehand_mask(image):
    plt.ioff()
    mask = {"data": None}

    fig, ax = plt.subplots()
    ax.imshow(image, cmap="gray")
    ax.set_title("Draw freehand tumor mask, then press Enter")
    ax.axis("off")

    h, w = image.shape
    yy, xx = np.mgrid[:h, :w]
    points = np.vstack((xx.ravel(), yy.ravel())).T

    def onselect(verts):
        path = MplPath(verts)
        inside = path.contains_points(points)
        mask["data"] = inside.reshape(h, w)

    def on_key(event):
        if event.key == "enter":
            plt.close(fig)

    lasso = LassoSelector(ax, onselect)
    _ = lasso
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show(block=True)

    if mask["data"] is None:
        raise RuntimeError("No mask drawn. Draw a region and press Enter.")

    return mask["data"]


def main():
    bbox_json_path = PART3_ARTIFACTS_DIR / "pet_registered_bbox.json"
    if not bbox_json_path.exists():
        print(f"Error: {bbox_json_path} not found. Run part3.py first.")
        return

    with open(bbox_json_path, "r") as f:
        bbox_data = json.load(f)

    z_index = int(bbox_data.get("slice_index", 0))
    mr_img = sitk.ReadImage(str(MR_PATH))
    mr_array = sitk.GetArrayFromImage(mr_img)
    if z_index >= mr_array.shape[0]:
        z_index = mr_array.shape[0] // 2
    mr_slice = mr_array[z_index]

    mask = _draw_freehand_mask(mr_slice)

    PART3_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    mask_path = PART3_ARTIFACTS_DIR / "mr_slice_gt_mask.npy"
    np.save(mask_path, mask.astype(np.uint8))

    meta = {
        "slice_index": int(z_index),
        "image_shape": [int(mr_slice.shape[0]), int(mr_slice.shape[1])],
        "mask_path": str(mask_path),
        "source": "MR",
    }
    meta_path = PART3_ARTIFACTS_DIR / "mr_slice_gt_mask.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved ground-truth mask to {mask_path}")
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()