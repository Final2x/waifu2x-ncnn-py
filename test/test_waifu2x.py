import sys
import os
import time
import numpy as np
import cv2
from skimage.metrics import structural_similarity
from PIL import Image

try:
    from waifu2x_ncnn_vulkan import Waifu2x
except ImportError:
    from waifu2x_ncnn_py import Waifu2x

print("System version: ", sys.version)


def calculate_image_similarity() -> bool:
    # Load the two images
    image1 = cv2.imread("./test.png")
    image2 = cv2.imread("./output.png")
    # Resize the two images to the same size
    height, width = image1.shape[:2]
    image2 = cv2.resize(image2, (width, height))
    # Convert the images to grayscale
    grayscale_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    grayscale_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # Calculate the Structural Similarity Index (SSIM) between the two images
    (score, diff) = structural_similarity(grayscale_image1, grayscale_image2, full=True)
    print("SSIM: {}".format(score))
    return score > 0.5


_gpuid = -1


def test_waifu2x():
    if _gpuid == -1:
        print("USE  ~~~~~~~~~~~~~~~~~CPU~~~~~~~~~~~~~~~~~~")
    else:
        print("USE  ~~~~~~~~~~~~~~~~~GPU~~~~~~~~~~~~~~~~~~")

    _scale = 2

    out_w = 0
    out_h = 0

    with Image.open("test.png") as image:
        out_w = image.width * _scale
        out_h = image.height * _scale
        waifu2x = Waifu2x(gpuid=_gpuid, scale=_scale, noise=0, model="models-upconv_7_anime_style_art_rgb")
        image = waifu2x.process_pil(image)
        image.save("output.png")

    with Image.open("output.png") as image:
        assert image.width == out_w
        assert image.height == out_h

    assert calculate_image_similarity()
