import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

from consts import MR_PATH, PART2_ARTIFACTS_DIR, PET_PATH
from dicom_io import (
    build_volume_4d,
    extract_basic,
    extract_frame_metadata,
    extract_spacing,
    load_dicom,
)
from gif_export import make_mip_gifs, save_gif


def pet_volume_for_registration(volume_4d, strategy="mean"):
    if strategy == "mid":
        return volume_4d[volume_4d.shape[0] // 2]
    if strategy == "mean":
        return np.mean(volume_4d, axis=0)
    raise ValueError("Unknown PET volume strategy")


def register_pet_to_mr(pet_img, mr_img):
    initial_transform = sitk.CenteredTransformInitializer(
        mr_img,
        pet_img,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    # Good parameters?
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.2)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0, minStep=1e-4, numberOfIterations=200
    )
    registration.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = registration.Execute(mr_img, pet_img)

    resampled_pet = sitk.Resample(
        pet_img, mr_img, final_transform, sitk.sitkLinear, 0.0, pet_img.GetPixelID()
    )
    return resampled_pet, final_transform


def normalize(img):
    p1, p99 = np.percentile(img, (1, 99))
    img = np.clip(img, p1, p99)
    return (img - p1) / (p99 - p1 + 1e-8)

def combine_images(mr_img, pet_img):
    # -------------------------
    # Visualization
    # -------------------------
    mr_np = sitk.GetArrayFromImage(mr_img)          # (z, y, x)
    pet_np = sitk.GetArrayFromImage(pet_img)  # (z, y, x)

    z = mr_np.shape[0] // 2
    mr_slice = mr_np[z]
    pet_slice = pet_np[z]

    mr_norm = normalize(mr_slice)
    pet_norm = normalize(pet_slice)

    alpha = 0.5 #is this good alpha?
    overlay = (1 - alpha) * mr_norm + alpha * pet_norm

    return overlay




def main():
    pet_dicom = load_dicom(PET_PATH)
    mr_img = sitk.ReadImage(str(MR_PATH))


    num_frames, rows, cols, pixel_array = extract_basic(pet_dicom)
    frame_times, frame_durations, frame_z = extract_frame_metadata(pet_dicom, num_frames)
    row_spacing, col_spacing, z_spacing = extract_spacing(pet_dicom)

    T = len(frame_times)
    Z = len(np.unique(frame_z))

    volume_4d = build_volume_4d(pixel_array, T, Z, rows, cols)
    volume_3d = pet_volume_for_registration(volume_4d, strategy="mean")

    pet_img = sitk.GetImageFromArray(volume_3d.astype(np.float32))
    pet_img.SetSpacing((col_spacing, row_spacing, z_spacing))

    print("MR dim:", mr_img.GetDimension(), "size:", mr_img.GetSize())
    print("PET dim:", pet_img.GetDimension(), "size:", pet_img.GetSize())

    # It throws a type mismatch erro otherwise. I assume on eof them is int and another is float
    pet_img = sitk.Cast(pet_img, sitk.sitkFloat32)
    mr_img = sitk.Cast(mr_img, sitk.sitkFloat32)

    resampled_pet, transform = register_pet_to_mr(pet_img, mr_img)

    overlay = combine_images(mr_img, resampled_pet)

    PART2_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    overlay_path = PART2_ARTIFACTS_DIR / "pet_mr_overlay.png"

    plt.figure(figsize=(6, 6))
    plt.imshow(overlay, cmap="gray")
    plt.title("PET (registered) over MR")
    plt.axis("off")
    plt.savefig(overlay_path, bbox_inches="tight", dpi=150)
    plt.close()

    print(f"Saved overlay image to {overlay_path}")

    mr_np = sitk.GetArrayFromImage(mr_img)
    pet_np = sitk.GetArrayFromImage(resampled_pet)
    mr_frames, pet_frames, fusion_frames = make_mip_gifs(mr_np, pet_np)
    mr_path = PART2_ARTIFACTS_DIR / "mr.gif"
    pet_path = PART2_ARTIFACTS_DIR / "pet.gif"
    fusion_path = PART2_ARTIFACTS_DIR / "fusion.gif"
    save_gif(mr_frames, mr_path, duration=0.02)
    save_gif(pet_frames, pet_path, duration=0.02)
    save_gif(fusion_frames, fusion_path, duration=0.02)



if __name__ == "__main__":
    main()