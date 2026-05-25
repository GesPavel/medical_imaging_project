import json

import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import numpy as np
import SimpleITK as sitk

from consts import PART2_ARTIFACTS_DIR, PART3_ARTIFACTS_DIR


def _select_bbox(image):
    # Ensure the UI stays open in IDEs that default to non-blocking show.
    plt.ioff()
    bbox = {"x_min": None, "y_min": None, "x_max": None, "y_max": None}

    fig, ax = plt.subplots()
    ax.imshow(image, cmap="gray")
    ax.set_title("Draw a rectangle around the tumor, then press Enter")
    ax.axis("off")

    def onselect(eclick, erelease):
        x0, y0 = eclick.xdata, eclick.ydata
        x1, y1 = erelease.xdata, erelease.ydata
        if x0 is None or y0 is None or x1 is None or y1 is None:
            return
        bbox["x_min"], bbox["x_max"] = sorted([float(x0), float(x1)])
        bbox["y_min"], bbox["y_max"] = sorted([float(y0), float(y1)])

    def on_key(event):
        if event.key == "enter":
            plt.close(fig)

    selector = RectangleSelector(
        ax,
        onselect,
        useblit=True,
        button=None,
        minspanx=5,
        minspany=5,
        spancoords="pixels",
        interactive=True,
    )
    # Keep a reference so the selector remains active.
    _ = selector
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show(block=True)

    if bbox["x_min"] is None:
        raise RuntimeError("No bounding box selected. Draw a rectangle and press Enter.")

    return bbox


def main():
    pet_path = PART2_ARTIFACTS_DIR / "resampled_pet.nrrd"
    if not pet_path.exists():
        raise FileNotFoundError(f"File not found: {pet_path}. Please run part2.py first.")

    pet_img = sitk.ReadImage(str(pet_path))
    volume_3d = sitk.GetArrayFromImage(pet_img)

    Z, rows, cols = volume_3d.shape
    z_mid = Z // 2
    img_slice = volume_3d[z_mid]

    bbox = _select_bbox(img_slice)

    output = {
        "source_file": pet_path.name,
        "slice_index": int(z_mid),
        "image_shape": [int(rows), int(cols)],
        "bbox": bbox,
        "bbox_width": float(bbox["x_max"] - bbox["x_min"]),
        "bbox_height": float(bbox["y_max"] - bbox["y_min"]),
    }

    PART3_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PART3_ARTIFACTS_DIR / "pet_registered_bbox.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"Saved bounding box to {output_path}.")


if __name__ == "__main__":
    main()
