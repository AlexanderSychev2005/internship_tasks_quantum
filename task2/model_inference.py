import os
from glob import glob

import cv2
from matplotlib import pyplot as plt

from algorithm_creation import (
    read_raster_image,
    match_loftr,
    draw_matches,
)

import argparse


def plot_images(image1, image2):
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(image1)
    plt.subplot(1, 2, 2)
    plt.imshow(image2)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./dataset",
        help="Path to the dataset with images",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.8,
        help="Confidence threshold for detections",
    )

    args = parser.parse_args()

    dataset_path = args.dataset_path
    confidence = args.confidence

    filenames = []

    for filename in glob(os.path.join(dataset_path, "*.jp2")):
        filenames.append(filename)

    image_indexes = [15, 5]

    images = []
    for i in image_indexes:
        image, meta = read_raster_image(filenames[i])
        images.append(image)
        print(f"Image {i} was created, shape: {image.shape}")

    dscale_images = [cv2.resize(image, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA) for image in images]

    plot_images(dscale_images[0], dscale_images[1])

    keypoints1, keypoints2, inliers, image1, image2 = match_loftr(
        images[0], images[1], (1098, 1098), confidence=confidence
    )

    num_keypoints = len(inliers)
    num_inliers = sum(inliers)[0]
    ratio = num_inliers / float(num_keypoints)

    print(
        f"Number of keypoints: {num_keypoints}, number of inliers: {num_inliers}, ratio: {ratio}"
    )

    output_figure = draw_matches(image1, image2, keypoints1, keypoints2, inliers)
    plt.show()
