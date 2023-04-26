"""
The MIT License (MIT)

Copyright (c) 2021 ArchieMeng

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
# 参考https://github.com/media2x/waifu2x-ncnn-vulkan-python，感谢原作者

import pathlib
import time
from PIL import Image
import numpy as np
import cv2

if __package__ or "." in __name__:
    from . import waifu2x_ncnn_vulkan_wrapper as wrapped
else:
    import waifu2x_ncnn_vulkan_wrapper as wrapped


class Waifu2x:
    """
    Waifu2x class which can do image super resolution.

    :param gpuid: gpu device to use, -1 for cpu
    :param tta_mode: enable test time argumentation
    :param num_threads: processing thread count
    :param noise: denoise level
    :param scale: upscale ratio
    :param tilesize: tile size
    :param model: waifu2x model name, can be "models-cunet", "models-upconv_7_anime_style_art_rgb" and "models-upconv_7_photo", default: models-cunet. Or an absolute path to a model folder
    """

    def __init__(
            self,
            gpuid: int = 0,
            tta_mode: bool = False,
            num_threads: int = 1,
            noise: int = 0,
            scale: int = 2,
            tilesize: int = 0,
            model: str = "models-cunet",
            **_kwargs,
    ):

        # check arguments' validity
        assert gpuid >= -1, "gpuid must >= -1"
        assert noise in range(-1, 4), "noise must be [-1, 3]"
        assert scale in range(1, 3), "scale must be 1 or 2"
        assert tilesize == 0 or tilesize >= 32, "tilesize must >= 32 or be 0"
        assert num_threads >= 1, "num_threads must be a positive integer"

        self._gpuid = gpuid

        self._waifu2x_object = wrapped.Waifu2xWrapped(gpuid, tta_mode, num_threads)

        self._model = model
        self._noise = noise
        self._scale = scale
        self._tilesize = tilesize

        self.set_parameters()

        self._load()

        self.raw_in_image = None
        self.raw_out_image = None

    def set_parameters(self, prepadding: int = 0) -> None:
        """
        Set parameters for Waifu2x

        :param prepadding: prepadding for waifu2x, default: 0, which means auto
        :return: None
        """

        self._waifu2x_object.set_parameters(self._noise, self._scale, prepadding, self._tilesize, self._model)

    def _load(
            self, param_path: pathlib.Path = None, model_path: pathlib.Path = None
    ) -> None:
        """
        Load models from given paths. Use self._model if one or all of the parameters are not given.

        :param param_path: the path to model params. usually ended with ".param"
        :param model_path: the path to model bin. usually ended with ".bin"
        :return: None
        """
        if param_path is None or model_path is None:
            model_path = pathlib.Path(self._model)
            if not model_path.is_dir():
                model_path = pathlib.Path(__file__).parent / "models" / self._model

            if self._noise == -1:
                param_path = model_path / "scale2.0x_model.param"
                model_path = model_path / "scale2.0x_model.bin"
                self._scale = 2
            elif self._scale == 1:
                param_path = model_path / f"noise{self._noise}_model.param"
                model_path = model_path / f"noise{self._noise}_model.bin"
            elif self._scale == 2:
                param_path = model_path / f"noise{self._noise}_scale2.0x_model.param"
                model_path = model_path / f"noise{self._noise}_scale2.0x_model.bin"

        if self._waifu2x_object.load(str(param_path), str(model_path)) != 0:
            raise Exception("Failed to load model")

    def process(self) -> None:
        if self._gpuid != -1:
            self._waifu2x_object.process(self.raw_in_image, self.raw_out_image)
        else:
            self._waifu2x_object.process_cpu(self.raw_in_image, self.raw_out_image)

    def process_pil(self, _image: Image) -> Image:
        """
        Process a PIL image

        :param _image: PIL image
        :return: processed PIL image
        """

        in_bytes = _image.tobytes()
        channels = int(len(in_bytes) / (_image.width * _image.height))
        out_bytes = (self._scale ** 2) * len(in_bytes) * b"\x00"

        self.raw_in_image = wrapped.Waifu2xImage(
            in_bytes,
            _image.width,
            _image.height,
            channels
        )

        self.raw_out_image = wrapped.Waifu2xImage(
            out_bytes,
            self._scale * _image.width,
            self._scale * _image.height,
            channels,
        )

        self.process()

        return Image.frombytes(
            _image.mode,
            (
                self._scale * _image.width,
                self._scale * _image.height,
            ),
            self.raw_out_image.get_data(),
        )

    def process_cv2(self, _image: np.ndarray) -> np.ndarray:
        """
        Process a cv2 image

        :param _image: cv2 image
        :return: processed cv2 image
        """
        _image = cv2.cvtColor(_image, cv2.COLOR_BGR2RGB)

        in_bytes = _image.tobytes()
        channels = int(len(in_bytes) / (_image.shape[1] * _image.shape[0]))
        out_bytes = (self._scale ** 2) * len(in_bytes) * b"\x00"

        self.raw_in_image = wrapped.Waifu2xImage(
            in_bytes,
            _image.shape[1],
            _image.shape[0],
            channels
        )

        self.raw_out_image = wrapped.Waifu2xImage(
            out_bytes,
            self._scale * _image.shape[1],
            self._scale * _image.shape[0],
            channels,
        )

        self.process()

        res = np.frombuffer(
            self.raw_out_image.get_data(),
            dtype=np.uint8
        ).reshape(
            self._scale * _image.shape[0],
            self._scale * _image.shape[1],
            channels
        )

        return cv2.cvtColor(res, cv2.COLOR_RGB2BGR)

    def process_bytes(self, _image_bytes: bytes, width: int, height: int, channels: int) -> bytes:
        """
        Process a bytes image, like bytes from ffmpeg

        :param _image_bytes: bytes
        :param width: image width
        :param height: image height
        :param channels: image channels
        :return: processed bytes image
        """
        if self.raw_in_image is None and self.raw_out_image is None:
            self.raw_in_image = wrapped.Waifu2xImage(
                _image_bytes,
                width,
                height,
                channels
            )

            self.raw_out_image = wrapped.Waifu2xImage(
                (self._scale ** 2) * len(_image_bytes) * b"\x00",
                self._scale * width,
                self._scale * height,
                channels,
            )

        self.raw_in_image.set_data(_image_bytes)

        self.process()

        return self.raw_out_image.get_data()


if __name__ == "__main__":
    waifu2x = Waifu2x(gpuid=0, scale=2, noise=3)

    time_start = time.time()

    with Image.open("input.jpg") as image:
        image = waifu2x.process_pil(image)
        image.save("output.jpg", quality=95)

    print(f"Time: {(time.time() - time_start) * 1000} ms")

    # test cv2

    time_start = time.time()

    image = cv2.imdecode(np.fromfile("input.jpg", dtype=np.uint8), cv2.IMREAD_COLOR)
    image = waifu2x.process_cv2(image)
    cv2.imencode(".jpg", image)[1].tofile("output_cv2.jpg")

    print(f"Time: {(time.time() - time_start) * 1000} ms")
