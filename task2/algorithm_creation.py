import cv2
import rasterio
import torch
from kornia_moons.viz import draw_LAF_matches
from rasterio.plot import reshape_as_image
import kornia as K
import kornia.feature as KF


def read_raster_image(image_path):
    with rasterio.open(image_path, "r", driver="JP2OpenJPEG") as src:
        raster_image = src.read()
        raster_meta = src.meta

    raster_image = reshape_as_image(raster_image)
    return raster_image, raster_meta


def prepare_image(image, device, image_size):
    image = K.utils.image_to_tensor(image)
    image = image.float().unsqueeze(dim=0).to(device) / 255.0

    image = K.geometry.resize(image, image_size, interpolation="area")

    return image


def match_loftr(image_path1, image_path2, image_size, pretrained="outdoor", confidence = 0.8):

    device = K.utils.get_cuda_device_if_available()
    image1 = prepare_image(image_path1, device, image_size=image_size)
    image2 = prepare_image(image_path2, device, image_size=image_size)

    matcher = KF.LoFTR(pretrained=pretrained)

    input_dict = {
        "image0": K.color.rgb_to_grayscale(
            image1
        ),  # LofTR works on grayscale images only
        "image1": K.color.rgb_to_grayscale(image2),
    }

    with torch.inference_mode():
        correspondences = matcher(input_dict)

    confidence_mask = correspondences["confidence"] > confidence
    indices = torch.nonzero(confidence_mask, as_tuple=True)

    keypoints1 = correspondences["keypoints0"][indices].cpu().numpy()
    keypoints2 = correspondences["keypoints1"][indices].cpu().numpy()

    Fm, inliers = cv2.findFundamentalMat(
        keypoints1, keypoints2, cv2.USAC_MAGSAC, 0.5, 0.999, 100000
    )
    inliers = inliers > 0

    return keypoints1, keypoints2, inliers, image1, image2


def draw_matches(image1, image2, keypoints1, keypoints2, inliers):
    output_figure = draw_LAF_matches(
        KF.laf_from_center_scale_ori(
            torch.from_numpy(keypoints1).view(1, -1, 2),
            torch.ones(keypoints1.shape[0]).view(1, -1, 1, 1),
            torch.ones(keypoints1.shape[0]).view(1, -1, 1),
        ),
        KF.laf_from_center_scale_ori(
            torch.from_numpy(keypoints2).view(1, -1, 2),
            torch.ones(keypoints2.shape[0]).view(1, -1, 1, 1),
            torch.ones(keypoints2.shape[0]).view(1, -1, 1),
        ),
        torch.arange(keypoints1.shape[0]).view(-1, 1).repeat(1, 2),
        K.tensor_to_image(image1),
        K.tensor_to_image(image2),
        inliers,
        draw_dict={
            "inlier_color": (0.2, 1, 0.2),
            "tentative_color": None,
            "feature_color": (0.2, 0.5, 1),
            "vertical": False,
        },
    )
    return output_figure
