import matplotlib.pyplot as plt
import numpy as np


def get_summary_images(volume_4d):
    T, Z, _, _ = volume_4d.shape
    t_mid = T // 2
    z_mid = Z // 2

    img_mid = volume_4d[t_mid, z_mid]
    img_last = volume_4d[-1, -1]
    img_avg = np.mean(volume_4d, axis=(0, 1))
    return img_mid, img_last, img_avg


def plot_summary_images(img_mid, img_last, img_avg):
    plt.figure()

    plt.subplot(1, 3, 1)
    plt.imshow(img_mid, cmap="gray")
    plt.title("middle")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(img_last, cmap="gray")
    plt.title("last frame")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(img_avg, cmap="gray")
    plt.title("average")
    plt.axis("off")

    plt.show()
