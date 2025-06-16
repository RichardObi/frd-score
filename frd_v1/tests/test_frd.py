""" Testing FRD metric.

run using pytest -rP tests/test_frd.py
"""
import logging
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import frd_v1.compute_frd as compute_frd

# Define allowed image extensions
IMAGE_EXTENSIONS = {
    "bmp",
    "jpg",
    "jpeg",
    "png",
    "tif",
    "tiff",
    "nii.gz",
}  # 'pgm', 'ppm', 'webp',

class TestFRDv1:
    def get_logger(self):
        LOGGING_LEVEL = logging.WARNING  # .INFO
        self.logger = logging.getLogger()  # (__name__)
        self.logger.setLevel(LOGGING_LEVEL)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(LOGGING_LEVEL)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

    def test_frd_2d(self):
        self.get_logger()
        self.logger.info("Testing frd_v1 metric for 2D images")
        path_a = "tmp_path1"
        path_b = "tmp_path2"

        # generate a few random images and test if frd calculation works for all allowed image file extensions
        # in_arr_1 = np.random.rand(128, 128, 3) * 255 # in case we want to try RGB images as well.
        in_arr_1 = np.random.rand(128, 128) * 255
        in_arr_2 = np.random.rand(128, 128) * 255

        in_image_1 = Image.fromarray(
            in_arr_1.astype(np.uint8), mode="L" # mode="RGB" # in case we want to try RGB images as well.
        )  # Testing grayscale image
        in_image_2 = Image.fromarray(
            in_arr_2.astype(np.uint8), mode="L"
        )  # Testing grayscale image

        # create folders for test
        Path(path_a).mkdir(exist_ok=True)
        Path(path_b).mkdir(exist_ok=True)

        for ext in IMAGE_EXTENSIONS:
            if ext != "nii.gz":
                # create image with different extensions
                in_image_1.save(f"{path_a}/img1.{ext}")
                in_image_1.save(f"{path_a}/img2.{ext}")
                in_image_2.save(f"{path_b}/img1.{ext}")
                in_image_2.save(f"{path_b}/img2.{ext}")

        # Ensure no empty datasets
        assert len(list(Path(path_a).glob("*"))) > 0
        assert len(list(Path(path_b).glob("*"))) > 0

        # Test if this function raises error
        try:
            frd_value = compute_frd.main(
                image_folder1=path_a,
                image_folder2=path_b,
                force_compute_fresh = True,
                interpret = False, # TODO: If True, TSNE() might throw an error depending on sklearn version - fix and test this as well
                parallelize = True,
            )
            self.logger.warning(
                f"FRD v1 value: {frd_value}"
            )
            print(f"FRD v1 value: {frd_value}")
        except Exception as e:
            raise e
        finally:
            os.system(f"rm -rf {path_a} {path_b} outputs")