import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import os
import math
sys.path.append("..")


def show_anns(anns, alpha=0.35):

    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate((np.random.random(3), [alpha]))
        img[m] = color_mask
    ax.imshow(img)


def show_mask(mask, ax, random_color=False):

    if random_color:
        color = np.concatenate((np.random.random(3), np.array([0.6])), axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):

    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):

    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def read_images_from_directory(directory_path):
    images = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            img_path = os.path.join(directory_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img_rgb)
    return images


def show_images_grid(images):
    # Determine grid size
    num_images = len(images)
    grid_size = math.ceil(math.sqrt(num_images))

    # Create figure and subplots
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(12, 12))

    # Flatten the axs array for easy indexing, in case grid is not a perfect square
    axs = axs.flatten()

    for i in range(num_images):
        axs[i].imshow(images[i], cmap='gray')
        axs[i].set_title(f'Image {i + 1}')
        axs[i].axis('off')  # Hide axes

    # Hide any remaining subplots
    for j in range(num_images, len(axs)):
        fig.delaxes(axs[j])

    # Adjust layout
    plt.tight_layout()
    plt.show()


def none_img(bg=None):
    # Create a black background image
    height, width = 100, 100  # You can adjust the size as needed
    if bg is None:
        bg = np.zeros((height, width, 3), dtype=np.uint8)

    # Define the text and its properties
    text = "None"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = bg.shape[1] / 100
    font_color = (255, 0, 0)  # Red color text
    thickness = round(2 * font_scale)
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

    # Calculate the text position so it's centered
    text_x = (bg.shape[1] - text_size[0]) // 2
    text_y = (bg.shape[0] + text_size[1]) // 2

    # Add the text to the black background
    cv2.putText(bg, text, (text_x, text_y), font, font_scale, font_color, thickness)

    return bg
