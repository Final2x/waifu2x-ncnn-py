import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
from waifu2x_ncnn_py import Waifu2x

print("System version: ", sys.version)

filePATH = Path(__file__).resolve().absolute()

print("filePATH: ", filePATH)


def calculate_image_similarity(image1: np.ndarray, image2: np.ndarray) -> bool:
    # Resize the two images to the same size
    height, width = image1.shape[:2]
    image2 = cv2.resize(image2, (width, height))
    # Convert the images to grayscale
    grayscale_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    grayscale_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # Calculate the Structural Similarity Index (SSIM) between the two images
    (score, diff) = structural_similarity(grayscale_image1, grayscale_image2, full=True)
    print("SSIM: {}".format(score))
    return bool(score > 0.5)


_gpuid = -1


def test_waifu2x() -> None:
    if _gpuid == -1:
        print("USE  ~~~~~~~~~~~~~~~~~CPU~~~~~~~~~~~~~~~~~~")
    else:
        print("USE  ~~~~~~~~~~~~~~~~~GPU~~~~~~~~~~~~~~~~~~")

    _scale = 2

    out_w = 0
    out_h = 0

    testimgPATH = filePATH.parent / "test.png"
    outputimgPATH = filePATH.parent / "output.png"

    with Image.open(str(testimgPATH)) as image:
        out_w = image.width * _scale
        out_h = image.height * _scale
        waifu2x = Waifu2x(gpuid=_gpuid, scale=_scale, noise=0, model="models-upconv_7_anime_style_art_rgb")
        image = waifu2x.process_pil(image)
        image.save(str(outputimgPATH))

    with Image.open(str(outputimgPATH)) as image:
        assert image.width == out_w
        assert image.height == out_h

    assert calculate_image_similarity(cv2.imread(str(testimgPATH)), cv2.imread(str(outputimgPATH)))
