import imageio
import numpy as np
from skimage.transform import resize
from scipy.ndimage import rotate
import matplotlib.pyplot as plt


def normalize(volume):
    p1, p99 = np.percentile(volume, (1, 99))
    volume = np.clip(volume, p1, p99)
    return (volume - p1) / (p99 - p1 + 1e-8)

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

def make_mip_gifs(mr_np, pet_np, alpha=0.4, n_angles=60, mr_cmap="gray", pet_cmap="magma"):
    """
    mr_np, pet_np: numpy arrays of shape (z, y, x), already co-registered
    out_dir: directory to save gifs
    """

    # Normalize ONCE per volume (important)
    mr_np = normalize(mr_np)
    pet_np = normalize(pet_np)

    mr_map = plt.get_cmap(mr_cmap)
    pet_map = plt.get_cmap(pet_cmap)

    angles = np.linspace(0, 180, n_angles)

    frames_mr = []
    frames_pet = []
    frames_fusion = []

    for angle in angles:
        # Rotate both volumes identically
        mr_rot = rotate(mr_np, angle, axes=(1, 2), reshape=False, order=1)
        pet_rot = rotate(pet_np, angle, axes=(1, 2), reshape=False, order=1)

        # MIP (front-view projection)
        mr_mip = np.max(mr_rot, axis=2)
        pet_mip = np.max(pet_rot, axis=2)

        # Flip to correct orientation
        mr_mip = np.flipud(mr_mip)
        pet_mip = np.flipud(pet_mip)

        mr_rgb = mr_map(mr_mip)[..., :3]
        pet_rgb = pet_map(pet_mip)[..., :3]

        # Fusion AFTER MIP
        fusion_rgb = (1 - alpha) * mr_rgb + alpha * pet_rgb

        frames_mr.append((mr_rgb * 255).astype(np.uint8))
        frames_pet.append((pet_rgb * 255).astype(np.uint8))
        frames_fusion.append((fusion_rgb * 255).astype(np.uint8))

    # Save GIFs
    return frames_mr, frames_pet, frames_fusion

def save_gif(frames, path, duration=0.2):
    imageio.mimsave(path, frames, duration=duration)
