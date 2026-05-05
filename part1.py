import numpy as np

from consts import PET_PATH, PART1_ARTIFACTS_DIR
from dicom_io import (
    build_volume_4d,
    extract_basic,
    extract_frame_metadata,
    extract_spacing,
    load_dicom,
)
from gif_export import build_gif_frames, save_gif
from plotting import get_summary_images, plot_summary_images


def main():
    ds = load_dicom(PET_PATH)

    num_frames, rows, cols, pixel_array = extract_basic(ds)
    frame_times, frame_durations, frame_z = extract_frame_metadata(ds, num_frames)
    row_spacing, col_spacing, z_spacing = extract_spacing(ds)

    T = len(frame_times)
    Z = len(np.unique(frame_z))

    volume_4d = build_volume_4d(pixel_array, T, Z, rows, cols)

    print("volume shape:", volume_4d.shape)

    img_mid, img_last, img_avg = get_summary_images(volume_4d)
    plot_summary_images(img_mid, img_last, img_avg)

    frames_axial, frames_coronal, frames_sagittal = build_gif_frames(
        volume_4d, row_spacing, col_spacing, z_spacing
    )

    dir = PART1_ARTIFACTS_DIR
    dir.mkdir(parents=True, exist_ok=True)
    save_gif(frames_axial, PART1_ARTIFACTS_DIR / "axial.gif")
    save_gif(frames_coronal, PART1_ARTIFACTS_DIR / "coronal.gif")
    save_gif(frames_sagittal, PART1_ARTIFACTS_DIR / "sagittal.gif")

    print(f"Saved GIFs to {dir}.")


if __name__ == "__main__":
    main()
