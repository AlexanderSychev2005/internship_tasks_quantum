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
    """
    Plots two images side by side for comparison.
    """
    plt.figure()
    plt.suptitle("Input images")

    plt.subplot(1, 2, 1)
    plt.imshow(image1)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(image2)
    plt.axis("off")

    plt.show()

def make_prediction(image1, image2, image_size, confidence, pretrained):
    """
    Makes keypoint matching prediction between two images using LoFTR.
    """
    keypoints1, keypoints2, inliers, img1, img2 = match_loftr(
        image1, image2, image_size, confidence=confidence, pretrained=pretrained
    )
    return keypoints1, keypoints2, inliers, img1, img2



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

    parser.add_argument(
        "--image_size",
        nargs=2,
        type=int,
        default=[1098, 1098],
        help="Size to which images are resized for model inference",
    )

    parser.add_argument(
        "--indexes_of_images",
        nargs=2,
        type=int,
        default=[15, 5],
        help="Indexes of images to be processed from the dataset",
    )
    parser.add_argument(
        "--model_pretrained",
        type=str,
        default="outdoor",
        help="Pretrained weights for LoFTR model",
    )

    args = parser.parse_args()

    dataset_path = args.dataset_path
    confidence = args.confidence
    image_size = tuple(args.image_size)
    image_indexes = tuple(args.indexes_of_images)
    pretrained_model = args.model_pretrained

    filenames = []

    for filename in glob(os.path.join(dataset_path, "*.jp2")):
        filenames.append(filename)


    images = []
    for i in image_indexes:
        image, meta = read_raster_image(filenames[i])
        images.append(image)
        print(f"Image {i} was created, shape: {image.shape}")

    dscale_images = [cv2.resize(image, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA) for image in images]

    plot_images(dscale_images[0], dscale_images[1])

    keypoints1, keypoints2, inliers, image1, image2 = make_prediction(
        images[0], images[1], image_size, confidence, pretrained_model)

    num_keypoints = len(inliers)
    num_inliers = sum(inliers)[0]
    ratio = num_inliers / float(num_keypoints)

    print(
        f"Number of keypoints: {num_keypoints}, number of inliers: {num_inliers}, ratio: {ratio}"
    )

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set(xticks=[], yticks=[])
    ax.set_title(f"LoFTR matches with confidence > {confidence:.2f}, Inliers ratio: {ratio:.2f}")
    output_figure = draw_matches(ax, image1, image2, keypoints1, keypoints2, inliers)
    plt.show()

    plt.savefig("result.png")

