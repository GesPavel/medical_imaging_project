import numpy as np

from dicom_io import (
    build_volume_4d,
    extract_basic,
    extract_frame_metadata,
    extract_spacing,
    load_dicom,
)
from gif_export import build_gif_frames, save_gifs
from plotting import get_summary_images, plot_summary_images
from validation import (
    validate_block_structure,
    validate_frame_counts,
    validate_time_vectors,
    validate_z_consistency,
)


def main():
    ds = load_dicom("pet_study")

    num_frames, rows, cols, pixel_array = extract_basic(ds)
    frame_times, frame_durations, frame_z = extract_frame_metadata(ds, num_frames)
    row_spacing, col_spacing, z_spacing = extract_spacing(ds)

    T = len(frame_times)
    Z = len(np.unique(frame_z))

    validate_time_vectors(frame_times, frame_durations)
    validate_frame_counts(num_frames, T, Z)
    validate_block_structure(frame_z, T, Z)
    validate_z_consistency(frame_z, T, Z)

    volume_4d = build_volume_4d(pixel_array, T, Z, rows, cols)

    print("volume shape:", volume_4d.shape)

    img_mid, img_last, img_avg = get_summary_images(volume_4d)
    plot_summary_images(img_mid, img_last, img_avg)

    frames_axial, frames_coronal, frames_sagittal = build_gif_frames(
        volume_4d, row_spacing, col_spacing, z_spacing
    )
    save_gifs(frames_axial, frames_coronal, frames_sagittal, duration=0.2)

    print("Saved: axial.gif, coronal.gif, sagittal.gif")


if __name__ == "__main__":
    main()