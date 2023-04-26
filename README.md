# waifu2x-ncnn-py
Python Binding for waifu2x-ncnn-vulkan with PyBind11 [![PyPI version](https://badge.fury.io/py/waifu2x-ncnn-py.svg?123456)](https://badge.fury.io/py/waifu2x-ncnn-py?123456) [![test_pip](https://github.com/Tohrusky/waifu2x-ncnn-py/actions/workflows/test_pip.yml/badge.svg)](https://github.com/Tohrusky/waifu2x-ncnn-py/actions/workflows/test_pip.yml) [![Release](https://github.com/Tohrusky/waifu2x-ncnn-py/actions/workflows/Release.yml/badge.svg)](https://github.com/Tohrusky/waifu2x-ncnn-py/actions/workflows/Release.yml)

Image Super-Resolution for Anime-style art using Deep Convolutional Neural Networks. And it supports photo. This wrapper provides an easy-to-use interface for running the pre-trained Waifu2x model.

### Current building status matrix
| System        | Status                                                                                                                                                                                                                              | CPU (32bit)  |  CPU (64bit)       | GPU (32bit)  | GPU (64bit)        |
|:-------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------:|:------------------:|:------------:|:------------------:|
| Linux (Clang) | [![CI-Linux-x64-Clang](https://github.com/Tohrusky/waifu2x-ncnn-py/actions/workflows/CI-Linux-x64-Clang.yml/badge.svg)](https://github.com/Tohrusky/waifu2x-ncnn-py/actions/workflows/CI-Linux-x64-Clang.yml)                   | —            | :white_check_mark: | —            | :white_check_mark: |
| Linux (GCC)   | [![CI-Linux-x64-GCC](https://github.com/Tohrusky/waifu2x-ncnn-py/actions/workflows/CI-Linux-x64-GCC.yml/badge.svg)](https://github.com/Tohrusky/waifu2x-ncnn-py/actions/workflows/CI-Linux-x64-GCC.yml)                         | —            | :white_check_mark: | —            | :white_check_mark: |
| Windows       | [![CI-Windows-x64-MSVC](https://github.com/Tohrusky/waifu2x-ncnn-py/actions/workflows/CI-Windows-x64-MSVC.yml/badge.svg)](https://github.com/Tohrusky/waifu2x-ncnn-py/actions/workflows/CI-Windows-x64-MSVC.yml)                | —            | :white_check_mark: | —            | :white_check_mark: |
| MacOS         | [![CI-MacOS-Universal-Clang](https://github.com/Tohrusky/waifu2x-ncnn-py/actions/workflows/CI-MacOS-Universal-Clang.yml/badge.svg)](https://github.com/Tohrusky/waifu2x-ncnn-py/actions/workflows/CI-MacOS-Universal-Clang.yml) | —            | :white_check_mark: | —            | :white_check_mark: |
| MacOS (ARM)   | [![CI-MacOS-Universal-Clang](https://github.com/Tohrusky/waifu2x-ncnn-py/actions/workflows/CI-MacOS-Universal-Clang.yml/badge.svg)](https://github.com/Tohrusky/waifu2x-ncnn-py/actions/workflows/CI-MacOS-Universal-Clang.yml) | —            | :white_check_mark: | —            | :white_check_mark: |




# Usage
```Python >= 3.6 (>= 3.9 in MacOS arm)```

To use this package, simply install it via pip:
```sh
pip install waifu2x-ncnn-py
```
For Linux user:
```sh
apt install -y libomp5 libvulkan-dev
```
Then, import the Waifu2x class from the package:

```python
from waifu2x_ncnn_py import Waifu2x
```
To initialize the model:

```python
waifu2x = Waifu2x(gpuid: int = 0, tta_mode: bool = False, num_threads: int = 1, noise: int = 0, scale: int = 2, tilesize: int = 0, model: str = "models-cunet", **_kwargs)
# model can be "models-cunet", "models-upconv_7_anime_style_art_rgb" and "models-upconv_7_photo"
# or an absolute path to the models' directory
```
Here, gpuid specifies the GPU device to use (-1 means use CPU), tta_mode enables test-time augmentation, num_threads sets the number of threads for processing, noise specifies the level of noise to apply to the image (-1 to 3), scale is the scaling factor for super-resolution (1 to 4), tilesize specifies the tile size for processing (0 or >= 32), and model specifies the name of the pre-trained model to use.

Once the model is initialized, you can use the upscale method to super-resolve your images:

### Pillow
```python
from PIL import Image
waifu2x = Waifu2x(gpuid=0, scale=2, noise=3)
with Image.open("input.jpg") as image:
    image = waifu2x.process_pil(image)
    image.save("output.jpg", quality=95)
```

### opencv-python
```python
import cv2
waifu2x = Waifu2x(gpuid=0, scale=2, noise=3)
image = cv2.imdecode(np.fromfile("input.jpg", dtype=np.uint8), cv2.IMREAD_COLOR)
image = waifu2x.process_cv2(image)
cv2.imencode(".jpg", image)[1].tofile("output_cv2.jpg")
```

### ffmpeg
```python
import subprocess as sp
# your ffmpeg parameters
command_out = [FFMPEG_BIN,........] 
command_in = [FFMPEG_BIN,........]
pipe_out = sp.Popen(command_out, stdout=sp.PIPE, bufsize=10 ** 8)
pipe_in = sp.Popen(command_in, stdin=sp.PIPE)
waifu2x = Waifu2x(gpuid=0, scale=2, noise=3)
while True:
    raw_image = pipe_out.stdout.read(src_width * src_height * 3)
    if not raw_image:
        break
    raw_image = waifu2x.process_bytes(raw_image, src_width, src_height, 3)
    pipe_in.stdin.write(raw_image)
```
# Build
[here](https://github.com/Tohrusky/waifu2x-ncnn-py/blob/main/.github/workflows/Release.yml) 

*The project just only been tested in Ubuntu 18+ and Debian 9+ environments on Linux, so if the project does not work on your system, please try building it.*


# References
The following references were used in the development of this project:

[nihui/waifu2x-ncnn-vulkan](https://github.com/nihui/waifu2x-ncnn-vulkan) - This project was the main inspiration for our work. It provided the core implementation of the Waifu2x algorithm using the ncnn and Vulkan libraries.

[Waifu2x](https://github.com/nagadomi/waifu2x) - Waifu2x is an Image Super-Resolution algorithm for Anime-style art using Deep Convolutional Neural Networks. And it supports photo.

[media2x/waifu2x-ncnn-vulkan-python](https://github.com/media2x/waifu2x-ncnn-vulkan-python) - This project was used as a reference for implementing the wrapper. *Special thanks* to the original author for sharing the code. 

[ncnn](https://github.com/Tencent/ncnn) - ncnn is a high-performance neural network inference framework developed by Tencent AI Lab. 

# License
This project is licensed under the BSD 3-Clause - see the [LICENSE file](https://github.com/Tohrusky/realcugan-ncnn-py/blob/main/LICENSE) for details.
