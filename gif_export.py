import imageio
import numpy as np
from skimage.transform import resize


def norm_and_resize(img, scale_y=1.0, scale_x=1.0):
    img = img.astype(float)
    img -= img.min()
    if img.max() > 0:
        img /= img.max()
    img = (img * 255).astype(np.uint8)

    if scale_y != 1.0 or scale_x != 1.0:
        new_h = int(img.shape[0] * scale_y)
        new_w = int(img.shape[1] * scale_x)
        img = resize(img, (new_h, new_w), order=1, preserve_range=True).astype(np.uint8)

    return img


def build_gif_frames(volume_4d, row_spacing, col_spacing, z_spacing, max_frames=None):
    T, Z, rows, cols = volume_4d.shape
    z_mid = Z // 2
    y_mid = rows // 2
    x_mid = cols // 2

    frames_axial = []
    frames_coronal = []
    frames_sagittal = []

    total_frames = T if max_frames is None else min(T, max_frames)
    for t in range(total_frames):
        axial = volume_4d[t, z_mid]

        coronal = volume_4d[t, :, y_mid, :]
        coronal = np.flipud(coronal)

        sagittal = volume_4d[t, :, :, x_mid]
        sagittal = np.flipud(sagittal)

        frames_axial.append(norm_and_resize(axial))

        frames_coronal.append(
            norm_and_resize(
                coronal,
                scale_y=z_spacing / row_spacing,
                scale_x=1.0,
            )
        )

        frames_sagittal.append(
            norm_and_resize(
                sagittal,
                scale_y=z_spacing / col_spacing,
                scale_x=1.0,
            )
        )

    return frames_axial, frames_coronal, frames_sagittal


def save_gif(frames, path, duration=0.2):
    imageio.mimsave(path, frames, duration=duration)
