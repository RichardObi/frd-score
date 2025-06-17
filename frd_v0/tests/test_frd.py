""" Testing FRD metric.

run using pytest -rP tests/test_frd.py
"""
import logging
import os
import random
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

from frd_v0.src.frd_score import frd


class TestFRD:
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
        self.logger.info("Testing FRD metric for 2D data")
        path_a = "tmp_path1"
        path_b = "tmp_path2"

        # generate a few random images and test if frd calculation works for all allowed image file extensions
        in_arr_1 = np.random.rand(128, 128, 3) * 255
        in_arr_2 = np.random.rand(128, 128) * 255

        in_image_1 = Image.fromarray(
            in_arr_1.astype(np.uint8), mode="RGB"
        )  # Testing RGB image
        in_image_2 = Image.fromarray(
            in_arr_2.astype(np.uint8), mode="L"
        )  # Testing grayscale image

        # create folders for test
        Path(path_a).mkdir(exist_ok=True)
        Path(path_b).mkdir(exist_ok=True)

        for ext in frd.IMAGE_EXTENSIONS:
            if ext != "nii.gz":
                # create image with different extensions
                in_image_1.save(f"{path_a}/img.{ext}")
                in_image_2.save(f"{path_b}/img.{ext}")

        features = [
            "firstorder",
            "glcm",
            "glrlm",
            "gldm",
            "glszm",
            "ngtdm",
            "shape2D",
            "shape",
        ]
        norm_type = "zscore"
        norm_range = [0, 7.45670747756958]

        # Test if this function raises error
        try:
            paths = [path_a, f"{path_a}/statistics.npz"]
            frd.save_frd_stats(
                paths,
                features,
                norm_type,
                norm_range,
                paths_masks=None,
                resize_size=120,
                verbose=True,
                save_features=True,
            )

        except Exception as e:
            raise e
        # Test if this function raises error
        try:
            paths = [path_a, path_b]
            frd_value = frd.compute_frd(
                paths,
                features,
                norm_type,
                norm_range,
                paths_masks=None,
                resize_size=224,
                verbose=True,
                save_features=False,
                norm_sets_separately=False,
            )
            self.logger.warning(
                f"FRD value 2D no masks, zscore normalized: {frd_value}"
            )
            print(f"FRD value: {frd_value}")
        except Exception as e:
            raise e
        finally:
            os.system(f"rm -rf {path_a} {path_b}")

    def test_frd_3d(self):
        self.get_logger()
        self.logger.info("Testing FRD metric for 3D data")
        path_a = "tmp_path3"
        path_b = "tmp_path4"

        # generate a few random images and test if frd calculation works for all allowed image file extensions
        # generating an a and b version to have a bit of variation in the dataset
        in_arr_1_a = np.random.rand(64, 64, 20, 1) * 255
        in_arr_1_b = np.random.rand(64, 64, 20, 1) * 255
        in_arr_2_a = np.random.rand(64, 64, 20, 1) * 255
        in_arr_2_b = np.random.rand(64, 64, 20, 1) * 255

        in_image_1_a = nib.Nifti1Image(in_arr_1_a.astype(np.uint8), affine=np.eye(4))
        in_image_1_b = nib.Nifti1Image(in_arr_1_b.astype(np.uint8), affine=np.eye(4))
        in_image_2_a = nib.Nifti1Image(
            in_arr_2_a.astype(np.uint8), affine=np.eye(4)
        )  # Nifti2Image is a different class from Nifti1Image
        in_image_2_b = nib.Nifti1Image(
            in_arr_2_b.astype(np.uint8), affine=np.eye(4)
        )  # Nifti2Image is a different class from Nifti1Image

        # create folders for test
        Path(path_a).mkdir(exist_ok=True)
        Path(path_b).mkdir(exist_ok=True)

        for i in range(0, 10):
            # create image with different extensions
            if i % 2 == 0:
                nib.save(in_image_1_a, f"{path_a}/img{i}.nii.gz")
                nib.save(in_image_2_a, f"{path_b}/img{i}.nii.gz")
            else:
                nib.save(in_image_1_b, f"{path_a}/img{i}.nii.gz")
                nib.save(in_image_2_b, f"{path_b}/img{i}.nii.gz")

        features = [
            "firstorder",
            "glcm",
            "glrlm",
            "gldm",
            "glszm",
            "ngtdm",
            "shape",
            "shape2D",
        ]
        norm_type = "minmax"
        norm_range = [1, 30]
        # Test if this function raises error
        try:
            # Should get very high FRD value
            paths = [path_a, path_b]
            frd_value = frd.compute_frd(
                paths,
                features,
                norm_type,
                norm_range,
                paths_masks=None,
                resize_size=64,
                verbose=True,
                save_features=False,
                norm_sets_separately=False,
            )
            self.logger.warning(
                f"FRD value 3D no masks, minmax normalized: {frd_value}"
            )
            assert (
                frd_value != 0.0
            ), f"FRD 3D no masks should not be 0, as we are comparing different images (and different masks). Got: {frd_value}"
        except Exception as e:
            raise e
        try:
            # Now we should get a very low FRD value comparing tmp_path1 with tmp_path1
            paths = [path_a, path_a]
            norm_type = "minmax"
            norm_range = [0.0, 5.0]
            frd_value = frd.compute_frd(
                paths,
                features,
                norm_type,
                norm_range,
                paths_masks=None,
                resize_size=None,
                verbose=True,
                save_features=False,
                norm_sets_separately=False,
            )
            self.logger.warning(
                f"FRD value 3D no masks comparing identical datasets, minmax normalized: {frd_value}"
            )
            assert (
                frd_value < 0.001 and frd_value > -0.001
            ), f"FRD should be 0 or very close to 0, as we are comparing the same images. Got: {frd_value}"
        except Exception as e:
            raise e

        # Now we create a few random masks to test FDR generation with masks
        # generate a few random images and test if frd calculation works for all allowed image file extensions
        # Guarantee that the random mask is not too small, i.e. at least one pixel.
        rand_x = int(random.uniform(1, 64))
        rand_y = int(random.uniform(2, 64))
        rand_z = int(random.uniform(3, 20))
        in_mask_arr_1 = np.zeros_like(in_arr_1_a) * 255  # np.zeros(64, 64, 20, 1) * 255
        in_mask_arr_2 = np.zeros_like(in_arr_2_a) * 255  # np.zeros(64, 64, 20, 1) * 255

        in_mask_arr_1[:rand_x, :rand_y, :rand_z, :] = 1
        # different masks in in_mask_arr_1 and in_mask_arr_2
        in_mask_arr_2[rand_x:, rand_y:, rand_z:, :] = 1

        in_image_1 = nib.Nifti1Image(in_mask_arr_1.astype(np.uint8), affine=np.eye(4))
        in_image_2 = nib.Nifti1Image(
            in_mask_arr_2.astype(np.uint8), affine=np.eye(4)
        )  # Nifti2Image is a different class from Nifti1Image

        # create folders for test
        Path(f"{path_a}_mask").mkdir(exist_ok=True)
        Path(f"{path_b}_mask").mkdir(exist_ok=True)

        for i in range(0, 10):
            # create and store temporary nifti images
            nib.save(in_image_1, f"{path_a}_mask/mask{i}.nii.gz")
            nib.save(in_image_2, f"{path_b}_mask/mask{i}.nii.gz")

        ### Try same images but different masks
        paths_mask = [f"{path_a}_mask", f"{path_b}_mask"]
        norm_type = "minmax"
        norm_range = [0.0, 7.0]
        paths = [path_a, path_a]
        try:
            frd_value = frd.compute_frd(
                paths,
                features,
                norm_type,
                norm_range,
                paths_masks=paths_mask,
                resize_size=None,
                verbose=True,
                save_features=False,
                norm_sets_separately=True,
                num_workers=1,
            )
            self.logger.warning(
                f"FRD value 3D with masks (but same images), minmax normalized: {frd_value}"
            )
            assert (
                frd_value > 1.0
            ), f"FRD 3D with masks should be >1.0, as we are comparing different images (with different masks). Got: {frd_value}"
        except Exception as e:
            raise e

        ### Try different images with different masks
        paths = [path_a, path_b]
        try:
            frd_value = frd.compute_frd(
                paths,
                features,
                norm_type,
                norm_range,
                paths_masks=paths_mask,
                resize_size=None,
                verbose=True,
                save_features=False,
                norm_sets_separately=True,
            )
            self.logger.warning(
                f"FRD value 3D with masks (but same images), minmax normalized: {frd_value}"
            )
            assert (
                frd_value != 0.0
            ), f"FRD 3D with masks should not be 0, as we are comparing different images (and different masks). Got: {frd_value}"
        except Exception as e:
            raise e

        ### Try with paths providing list of image_paths instead of parent folder
        ## Also test if more masks are available than images.
        # paths = [f"{path_a}", [f"{path_b}/img{i}.nii.gz" for i in range(0, 10)]]
        paths = [
            [f"{path_a}/img{i}.nii.gz" for i in range(0, 5)],
            [f"{path_b}/img{i}.nii.gz" for i in range(0, 10)],
        ]

        # paths_mask = [f"{path_a}_mask", [f"{path_b}_mask/mask{i}.nii.gz" for i in range(0, 10)]]
        paths_mask = [
            [f"{path_b}_mask/mask{i}.nii.gz" for i in range(0, 8)],
            [f"{path_b}_mask/mask{i}.nii.gz" for i in range(0, 10)],
        ]

        try:
            # paths = [f"{path_a}", f"{path_b}"]
            # [f"{path_b}/img{i}.nii.gz" for i in range(0, 1)]]
            # [[f"{path_a}/img1.nii.gz", f"{path_a}/img2.nii.gz", f"{path_a}/img3.nii.gz"],
            #     [f"{path_b}/img1.nii.gz", f"{path_b}/img2.nii.gz", f"{path_b}/img3.nii.gz"]]
            # paths_mask = [[f"{path_a}_mask/mask{i}.nii.gz" for i in range(0, 10)],[f"{path_b}_mask/mask{i}.nii.gz" for i in range(0, 10)]]
            # paths_mask = [[f"{path_a}_mask/mask1.nii.gz", f"{path_a}_mask/mask2.nii.gz", f"{path_a}_mask/mask3.nii.gz"],
            #              [f"{path_b}_mask/mask1.nii.gz", f"{path_b}_mask/mask2.nii.gz", f"{path_b}_mask/mask3.nii.gz"]]
            frd_value = frd.compute_frd(
                paths,
                features,
                norm_type,
                norm_range,
                paths_masks=paths_mask,
                resize_size=None,
                verbose=True,
                save_features=False,
                norm_sets_separately=True,
            )
            self.logger.warning(
                f"FRD value 3D with masks (but same images), minmax normalized (image and mask paths provided explicitly): {frd_value}"
            )
            assert (
                frd_value != 0.0
            ), f"FRD 3D with masks should not be 0, as we are comparing different images (and different masks, image and mask paths provided explicitly). Got: {frd_value}"
        except Exception as e:
            raise e
        finally:
            os.system(f"rm -rf {path_a} {path_b} {path_a}_mask {path_b}_mask")
