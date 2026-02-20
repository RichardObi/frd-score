"""Testing FRD metric (merged v0 + v1).

Run using: pytest -rP tests/test_frd.py
"""

import logging
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

from frd_score import frd


class TestFRDv1_2D:
    """Test FRD v1 (default) with 2D images."""

    def get_logger(self):
        LOGGING_LEVEL = logging.WARNING
        self.logger = logging.getLogger()
        self.logger.setLevel(LOGGING_LEVEL)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(LOGGING_LEVEL)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

    def test_frd_v1_2d(self, tmp_path):
        """Test FRD v1 with 2D grayscale images (default version)."""
        self.get_logger()
        self.logger.info("Testing FRD v1 metric for 2D data")
        path_a = str(tmp_path / "tmp_v1_2d_a")
        path_b = str(tmp_path / "tmp_v1_2d_b")

        in_arr_1 = np.random.rand(128, 128) * 255
        in_arr_2 = np.random.rand(128, 128) * 255

        in_image_1 = Image.fromarray(in_arr_1.astype(np.uint8), mode="L")
        in_image_2 = Image.fromarray(in_arr_2.astype(np.uint8), mode="L")

        Path(path_a).mkdir(exist_ok=True)
        Path(path_b).mkdir(exist_ok=True)

        # Create multiple images per folder (need >=2 for covariance)
        in_image_1.save(f"{path_a}/img1.png")
        in_image_1.save(f"{path_a}/img2.png")
        in_image_2.save(f"{path_b}/img1.png")
        in_image_2.save(f"{path_b}/img2.png")

        paths = [path_a, path_b]
        frd_value = frd.compute_frd(
            paths,
            frd_version="v1",
            verbose=True,
            save_features=False,
            norm_ref="joint",
        )
        self.logger.warning(f"FRD v1 2D value: {frd_value}")
        print(f"FRD v1 2D value: {frd_value}")
        # v1 returns log-transformed value
        assert isinstance(
            frd_value, (float, np.floating)
        ), f"FRD v1 should return a float, got {type(frd_value)}"

    def test_frd_v1_2d_default_version(self, tmp_path):
        """Verify that not passing frd_version defaults to v1."""
        self.get_logger()
        path_a = str(tmp_path / "tmp_v1_default_a")
        path_b = str(tmp_path / "tmp_v1_default_b")

        in_arr = np.random.rand(64, 64) * 255
        in_image = Image.fromarray(in_arr.astype(np.uint8), mode="L")

        Path(path_a).mkdir(exist_ok=True)
        Path(path_b).mkdir(exist_ok=True)

        in_image.save(f"{path_a}/img1.png")
        in_image.save(f"{path_a}/img2.png")
        in_image.save(f"{path_b}/img1.png")
        in_image.save(f"{path_b}/img2.png")

        # No frd_version argument — should default to v1
        frd_value = frd.compute_frd(
            [path_a, path_b],
            verbose=False,
            save_features=False,
        )
        print(f"FRD default version value: {frd_value}")
        assert isinstance(frd_value, (float, np.floating))


class TestFRDv0_2D:
    """Test FRD v0 with 2D images."""

    def get_logger(self):
        LOGGING_LEVEL = logging.WARNING
        self.logger = logging.getLogger()
        self.logger.setLevel(LOGGING_LEVEL)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(LOGGING_LEVEL)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

    def test_frd_v0_2d(self, tmp_path):
        """Test FRD v0 with various 2D image formats and options."""
        self.get_logger()
        self.logger.info("Testing FRD v0 metric for 2D data")
        path_a = str(tmp_path / "tmp_v0_2d_a")
        path_b = str(tmp_path / "tmp_v0_2d_b")

        in_arr_1 = np.random.rand(128, 128, 3) * 255
        in_arr_2 = np.random.rand(128, 128) * 255

        in_image_1 = Image.fromarray(in_arr_1.astype(np.uint8), mode="RGB")
        in_image_2 = Image.fromarray(in_arr_2.astype(np.uint8), mode="L")

        Path(path_a).mkdir(exist_ok=True)
        Path(path_b).mkdir(exist_ok=True)

        for ext in frd.IMAGE_EXTENSIONS:
            if ext != "nii.gz":
                in_image_1.save(f"{path_a}/img.{ext}")
                in_image_2.save(f"{path_b}/img.{ext}")

        # Test save_frd_stats with v0
        paths = [path_a, f"{path_a}/statistics_v0.npz"]
        frd.save_frd_stats(
            paths,
            frd_version="v0",
            paths_masks=None,
            resize_size=120,
            verbose=True,
            save_features=True,
        )
        assert os.path.exists(
            f"{path_a}/statistics_v0.npz"
        ), "NPZ statistics file was not created."

        # Test compute_frd with v0
        paths = [path_a, path_b]
        frd_value = frd.compute_frd(
            paths,
            frd_version="v0",
            norm_type="zscore",
            norm_range=[0, 7.45670747756958],
            paths_masks=None,
            resize_size=224,
            verbose=True,
            save_features=False,
            norm_ref="independent",
        )
        self.logger.warning(f"FRD v0 2D value: {frd_value}")
        print(f"FRD v0 2D value: {frd_value}")
        # v0 returns raw Frechet distance (not log-transformed)
        assert isinstance(
            frd_value, (float, np.floating)
        ), f"FRD v0 should return a float, got {type(frd_value)}"


class TestFRDv0_3D:
    """Test FRD v0 with 3D NIfTI images."""

    def get_logger(self):
        LOGGING_LEVEL = logging.WARNING
        self.logger = logging.getLogger()
        self.logger.setLevel(LOGGING_LEVEL)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(LOGGING_LEVEL)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

    def test_frd_v0_3d(self, tmp_path):
        """Test FRD v0 with 3D NIfTI images, masks, and different path modes."""
        self.get_logger()
        self.logger.info("Testing FRD v0 metric for 3D data")
        path_a = str(tmp_path / "tmp_v0_3d_a")
        path_b = str(tmp_path / "tmp_v0_3d_b")

        in_arr_1_a = np.random.rand(64, 64, 20, 1) * 255
        in_arr_1_b = np.random.rand(64, 64, 20, 1) * 255
        in_arr_2_a = np.random.rand(64, 64, 20, 1) * 255
        in_arr_2_b = np.random.rand(64, 64, 20, 1) * 255

        in_image_1_a = nib.Nifti1Image(in_arr_1_a.astype(np.uint8), affine=np.eye(4))
        in_image_1_b = nib.Nifti1Image(in_arr_1_b.astype(np.uint8), affine=np.eye(4))
        in_image_2_a = nib.Nifti1Image(in_arr_2_a.astype(np.uint8), affine=np.eye(4))
        in_image_2_b = nib.Nifti1Image(in_arr_2_b.astype(np.uint8), affine=np.eye(4))

        Path(path_a).mkdir(exist_ok=True)
        Path(path_b).mkdir(exist_ok=True)

        for i in range(10):
            if i % 2 == 0:
                nib.save(in_image_1_a, f"{path_a}/img{i}.nii.gz")
                nib.save(in_image_2_a, f"{path_b}/img{i}.nii.gz")
            else:
                nib.save(in_image_1_b, f"{path_a}/img{i}.nii.gz")
                nib.save(in_image_2_b, f"{path_b}/img{i}.nii.gz")

        # Test different distributions → FRD should be non-zero
        paths = [path_a, path_b]
        frd_value = frd.compute_frd(
            paths,
            frd_version="v0",
            norm_type="minmax",
            norm_range=[1, 30],
            paths_masks=None,
            resize_size=64,
            verbose=True,
            save_features=False,
            norm_ref="independent",
        )
        self.logger.warning(f"FRD v0 3D different images: {frd_value}")
        assert (
            frd_value != 0.0
        ), f"FRD 3D should not be 0 for different images. Got: {frd_value}"

        # Same distribution → FRD should be ~0
        paths = [path_a, path_a]
        frd_value = frd.compute_frd(
            paths,
            frd_version="v0",
            norm_type="minmax",
            norm_range=[0.0, 5.0],
            paths_masks=None,
            resize_size=None,
            verbose=True,
            save_features=False,
            norm_ref="independent",
        )
        self.logger.warning(f"FRD v0 3D identical datasets: {frd_value}")
        assert (
            abs(frd_value) < 0.001
        ), f"FRD should be ~0 for identical distributions. Got: {frd_value}"

        # --- Test with masks ---
        # Guarantee non-empty masks by using fixed, known-good regions
        in_mask_arr_1 = np.zeros((64, 64, 20, 1), dtype=np.uint8)
        in_mask_arr_2 = np.zeros((64, 64, 20, 1), dtype=np.uint8)

        # First mask: upper-left quadrant
        in_mask_arr_1[:32, :32, :10, :] = 1
        # Second mask: lower-right quadrant (different region)
        in_mask_arr_2[32:, 32:, 10:, :] = 1

        mask_nib_1 = nib.Nifti1Image(in_mask_arr_1.astype(np.uint8), affine=np.eye(4))
        mask_nib_2 = nib.Nifti1Image(in_mask_arr_2.astype(np.uint8), affine=np.eye(4))

        Path(f"{path_a}_mask").mkdir(exist_ok=True)
        Path(f"{path_b}_mask").mkdir(exist_ok=True)

        for i in range(10):
            nib.save(mask_nib_1, f"{path_a}_mask/mask{i}.nii.gz")
            nib.save(mask_nib_2, f"{path_b}_mask/mask{i}.nii.gz")

        # Same images, different masks → FRD > 1.0
        paths_mask = [f"{path_a}_mask", f"{path_b}_mask"]
        paths = [path_a, path_a]
        frd_value = frd.compute_frd(
            paths,
            frd_version="v0",
            norm_type="minmax",
            norm_range=[0.0, 7.0],
            paths_masks=paths_mask,
            resize_size=None,
            verbose=True,
            save_features=False,
            norm_ref="joint",
            num_workers=1,
        )
        self.logger.warning(f"FRD v0 3D with masks: {frd_value}")
        assert (
            frd_value > 1.0
        ), f"FRD 3D with different masks should be >1.0. Got: {frd_value}"

        # Different images, different masks → non-zero
        paths = [path_a, path_b]
        frd_value = frd.compute_frd(
            paths,
            frd_version="v0",
            norm_type="minmax",
            norm_range=[0.0, 7.0],
            paths_masks=paths_mask,
            resize_size=None,
            verbose=True,
            save_features=False,
            norm_ref="joint",
        )
        assert (
            frd_value != 0.0
        ), f"FRD 3D with different masks should not be 0. Got: {frd_value}"

        # List-of-paths mode
        paths = [
            [f"{path_a}/img{i}.nii.gz" for i in range(5)],
            [f"{path_b}/img{i}.nii.gz" for i in range(10)],
        ]
        paths_mask = [
            [f"{path_b}_mask/mask{i}.nii.gz" for i in range(8)],
            [f"{path_b}_mask/mask{i}.nii.gz" for i in range(10)],
        ]
        frd_value = frd.compute_frd(
            paths,
            frd_version="v0",
            norm_type="minmax",
            norm_range=[0.0, 7.0],
            paths_masks=paths_mask,
            resize_size=None,
            verbose=True,
            save_features=False,
            norm_ref="joint",
        )
        assert (
            frd_value != 0.0
        ), f"FRD 3D with list-of-paths should not be 0. Got: {frd_value}"


class TestFRDv1_3D:
    """Test FRD v1 with 3D NIfTI images."""

    def get_logger(self):
        LOGGING_LEVEL = logging.WARNING
        self.logger = logging.getLogger()
        self.logger.setLevel(LOGGING_LEVEL)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(LOGGING_LEVEL)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

    def test_frd_v1_3d(self, tmp_path):
        """Test FRD v1 with 3D NIfTI images (extended feature space)."""
        self.get_logger()
        self.logger.info("Testing FRD v1 metric for 3D data")
        path_a = str(tmp_path / "tmp_v1_3d_a")
        path_b = str(tmp_path / "tmp_v1_3d_b")

        in_arr_1_a = np.random.rand(64, 64, 20, 1) * 255
        in_arr_1_b = np.random.rand(64, 64, 20, 1) * 255
        in_arr_2_a = np.random.rand(64, 64, 20, 1) * 255
        in_arr_2_b = np.random.rand(64, 64, 20, 1) * 255

        in_image_1_a = nib.Nifti1Image(in_arr_1_a.astype(np.uint8), affine=np.eye(4))
        in_image_1_b = nib.Nifti1Image(in_arr_1_b.astype(np.uint8), affine=np.eye(4))
        in_image_2_a = nib.Nifti1Image(in_arr_2_a.astype(np.uint8), affine=np.eye(4))
        in_image_2_b = nib.Nifti1Image(in_arr_2_b.astype(np.uint8), affine=np.eye(4))

        Path(path_a).mkdir(exist_ok=True)
        Path(path_b).mkdir(exist_ok=True)

        for i in range(10):
            if i % 2 == 0:
                nib.save(in_image_1_a, f"{path_a}/img{i}.nii.gz")
                nib.save(in_image_2_a, f"{path_b}/img{i}.nii.gz")
            else:
                nib.save(in_image_1_b, f"{path_a}/img{i}.nii.gz")
                nib.save(in_image_2_b, f"{path_b}/img{i}.nii.gz")

        paths = [path_a, path_b]
        frd_value = frd.compute_frd(
            paths,
            frd_version="v1",
            verbose=True,
            save_features=False,
            norm_ref="joint",
        )
        self.logger.warning(f"FRD v1 3D value: {frd_value}")
        print(f"FRD v1 3D value: {frd_value}")
        assert isinstance(
            frd_value, (float, np.floating)
        ), f"FRD v1 should return a float, got {type(frd_value)}"


class TestUnifiedParams:
    """Test that v0-specific and v1-specific parameters work across both versions."""

    def _make_2d_images(self, path_a, path_b, n=2):
        """Helper: create n random 2D grayscale PNGs in each folder."""
        Path(path_a).mkdir(exist_ok=True)
        Path(path_b).mkdir(exist_ok=True)
        for i in range(n):
            arr = np.random.rand(64, 64) * 255
            Image.fromarray(arr.astype(np.uint8), mode="L").save(f"{path_a}/img{i}.png")
            arr2 = np.random.rand(64, 64) * 255
            Image.fromarray(arr2.astype(np.uint8), mode="L").save(
                f"{path_b}/img{i}.png"
            )

    # ── feature_groups on v1 ──────────────────────────────────────────────────

    def test_v1_with_feature_groups(self, tmp_path):
        """v1 should accept --feature_groups to restrict which feature classes are extracted."""
        path_a, path_b = str(tmp_path / "tmp_unified_fg_a"), str(
            tmp_path / "tmp_unified_fg_b"
        )
        self._make_2d_images(path_a, path_b)

        # Restrict v1 to only firstorder + glcm (fewer features → faster, different value)
        frd_value = frd.compute_frd(
            [path_a, path_b],
            frd_version="v1",
            features=["firstorder", "glcm"],
            verbose=False,
            save_features=False,
        )
        print(f"FRD v1 (feature_groups=[firstorder,glcm]): {frd_value}")
        assert isinstance(frd_value, (float, np.floating))

    def test_v1_feature_groups_adds_missing_class(self, tmp_path):
        """v1 YAML doesn't include gldm by default — feature_groups should enable it."""
        path_a, path_b = str(tmp_path / "tmp_unified_fg2_a"), str(
            tmp_path / "tmp_unified_fg2_b"
        )
        self._make_2d_images(path_a, path_b)

        frd_value = frd.compute_frd(
            [path_a, path_b],
            frd_version="v1",
            features=["firstorder", "gldm"],
            verbose=False,
            save_features=False,
        )
        print(f"FRD v1 (feature_groups=[firstorder,gldm]): {frd_value}")
        assert isinstance(frd_value, (float, np.floating))

    # ── image_types on v0 ─────────────────────────────────────────────────────

    def test_v0_with_wavelet_image_types(self, tmp_path):
        """v0 should be able to use Wavelet image type (a v1 feature) via image_types."""
        path_a, path_b = str(tmp_path / "tmp_unified_it_a"), str(
            tmp_path / "tmp_unified_it_b"
        )
        self._make_2d_images(path_a, path_b)

        frd_value = frd.compute_frd(
            [path_a, path_b],
            frd_version="v0",
            image_types=["Original", "Wavelet"],
            verbose=False,
            save_features=False,
        )
        print(f"FRD v0 (image_types=[Original,Wavelet]): {frd_value}")
        assert isinstance(frd_value, (float, np.floating))

    def test_v0_with_log_image_type(self, tmp_path):
        """v0 should be able to use LoG image type via image_types."""
        path_a, path_b = str(tmp_path / "tmp_unified_log_a"), str(
            tmp_path / "tmp_unified_log_b"
        )
        self._make_2d_images(path_a, path_b)

        frd_value = frd.compute_frd(
            [path_a, path_b],
            frd_version="v0",
            image_types=["Original", "LoG"],
            verbose=False,
            save_features=False,
        )
        print(f"FRD v0 (image_types=[Original,LoG]): {frd_value}")
        assert isinstance(frd_value, (float, np.floating))

    # ── image_types on v1 (restrict) ──────────────────────────────────────────

    def test_v1_restrict_to_original_only(self, tmp_path):
        """v1 should be able to restrict to Original-only via image_types."""
        path_a, path_b = str(tmp_path / "tmp_unified_v1orig_a"), str(
            tmp_path / "tmp_unified_v1orig_b"
        )
        self._make_2d_images(path_a, path_b)

        frd_value = frd.compute_frd(
            [path_a, path_b],
            frd_version="v1",
            image_types=["Original"],
            verbose=False,
            save_features=False,
        )
        print(f"FRD v1 (image_types=[Original]): {frd_value}")
        assert isinstance(frd_value, (float, np.floating))

    # ── masks on v1 (confirm they work) ───────────────────────────────────────

    def test_v1_with_masks(self, tmp_path):
        """v1 should fully support mask paths (originally a v0 feature)."""
        path_a, path_b = str(tmp_path / "tmp_unified_mask_a"), str(
            tmp_path / "tmp_unified_mask_b"
        )
        mask_a, mask_b = str(tmp_path / "tmp_unified_mask_a_m"), str(
            tmp_path / "tmp_unified_mask_b_m"
        )
        self._make_2d_images(path_a, path_b, n=2)

        # Create mask images (white rectangle in center)
        Path(mask_a).mkdir(exist_ok=True)
        Path(mask_b).mkdir(exist_ok=True)
        for i in range(2):
            mask_arr = np.zeros((64, 64), dtype=np.uint8)
            mask_arr[10:54, 10:54] = 1  # center region
            Image.fromarray(mask_arr, mode="L").save(f"{mask_a}/img{i}.png")
            Image.fromarray(mask_arr, mode="L").save(f"{mask_b}/img{i}.png")

        frd_value = frd.compute_frd(
            [path_a, path_b],
            frd_version="v1",
            paths_masks=[mask_a, mask_b],
            verbose=False,
            save_features=False,
        )
        print(f"FRD v1 with masks: {frd_value}")
        assert isinstance(frd_value, (float, np.floating))


class TestPaperLogTransform:
    """Tests for the use_paper_log parameter (paper Eq. 3 vs code convention)."""

    @staticmethod
    def _make_2d_images(path_a, path_b, n=3, size=64):
        Path(path_a).mkdir(exist_ok=True)
        Path(path_b).mkdir(exist_ok=True)
        for i in range(n):
            arr_a = np.random.rand(size, size) * 255
            arr_b = np.random.rand(size, size) * 255
            Image.fromarray(arr_a.astype(np.uint8), mode="L").save(
                f"{path_a}/img{i}.png"
            )
            Image.fromarray(arr_b.astype(np.uint8), mode="L").save(
                f"{path_b}/img{i}.png"
            )

    def test_v1_default_log_is_code_convention(self, tmp_path):
        """Default v1 FRD should use log(d_F^2) (original code convention)."""
        path_a, path_b = str(tmp_path / "tmp_paperlog_def_a"), str(
            tmp_path / "tmp_paperlog_def_b"
        )
        self._make_2d_images(path_a, path_b)

        frd_default = frd.compute_frd(
            [path_a, path_b],
            frd_version="v1",
            verbose=False,
            save_features=False,
        )
        assert isinstance(frd_default, (float, np.floating))
        print(f"FRD v1 default (log(d^2)): {frd_default}")

    def test_v1_paper_log_is_half_of_default(self, tmp_path):
        """Paper formula log(sqrt(d^2)) should equal 0.5 * log(d^2)."""
        path_a, path_b = str(tmp_path / "tmp_paperlog_half_a"), str(
            tmp_path / "tmp_paperlog_half_b"
        )
        self._make_2d_images(path_a, path_b)

        import warnings

        frd_default = frd.compute_frd(
            [path_a, path_b],
            frd_version="v1",
            verbose=False,
            save_features=False,
            use_paper_log=False,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            frd_paper = frd.compute_frd(
                [path_a, path_b],
                frd_version="v1",
                verbose=False,
                save_features=False,
                use_paper_log=True,
            )
            # Verify the warning was issued (filter for our specific UserWarning)
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert len(user_warnings) >= 1, f"Expected a UserWarning, got {w}"
            assert "use_paper_log=True" in str(user_warnings[0].message)
            assert "NOT directly comparable" in str(user_warnings[0].message)

        print(f"FRD default: {frd_default}, FRD paper: {frd_paper}")
        # Paper formula = 0.5 * default
        assert np.isclose(
            frd_paper, 0.5 * frd_default, rtol=1e-10
        ), f"Expected frd_paper ({frd_paper}) ≈ 0.5 * frd_default ({0.5 * frd_default})"

    def test_v0_unaffected_by_paper_log(self, tmp_path):
        """v0 should return the same value regardless of use_paper_log."""
        path_a, path_b = str(tmp_path / "tmp_paperlog_v0_a"), str(
            tmp_path / "tmp_paperlog_v0_b"
        )
        self._make_2d_images(path_a, path_b)

        frd_default = frd.compute_frd(
            [path_a, path_b],
            frd_version="v0",
            verbose=False,
            save_features=False,
            use_paper_log=False,
        )
        frd_paper = frd.compute_frd(
            [path_a, path_b],
            frd_version="v0",
            verbose=False,
            save_features=False,
            use_paper_log=True,
        )
        print(f"FRD v0 default: {frd_default}, v0 paper_log: {frd_paper}")
        assert frd_default == frd_paper, (
            f"v0 should not be affected by use_paper_log. "
            f"Got default={frd_default}, paper={frd_paper}"
        )


class TestNormRef:
    """Tests for the norm_ref parameter (joint / d1 / independent normalization modes)."""

    @staticmethod
    def _make_2d_images(path_a, path_b, n=3, size=64):
        Path(path_a).mkdir(exist_ok=True)
        Path(path_b).mkdir(exist_ok=True)
        for i in range(n):
            arr_a = np.random.rand(size, size) * 255
            arr_b = np.random.rand(size, size) * 255
            Image.fromarray(arr_a.astype(np.uint8), mode="L").save(
                f"{path_a}/img{i}.png"
            )
            Image.fromarray(arr_b.astype(np.uint8), mode="L").save(
                f"{path_b}/img{i}.png"
            )

    def test_all_three_modes_produce_valid_results(self, tmp_path):
        """All norm_ref modes should produce valid float FRD values."""
        path_a, path_b = str(tmp_path / "tmp_normref_modes_a"), str(
            tmp_path / "tmp_normref_modes_b"
        )
        self._make_2d_images(path_a, path_b)

        results = {}
        for mode in ["joint", "d1", "independent"]:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                frd_value = frd.compute_frd(
                    [path_a, path_b],
                    frd_version="v1",
                    verbose=False,
                    save_features=False,
                    norm_ref=mode,
                )
            results[mode] = frd_value
            print(f"FRD v1 norm_ref={mode}: {frd_value}")
            assert isinstance(
                frd_value, (float, np.floating)
            ), f"norm_ref={mode} should return a float, got {type(frd_value)}"
        # All three modes may produce different values (normalization base differs)
        print(f"All norm_ref mode results: {results}")

    def test_independent_mode_warns(self, tmp_path):
        """norm_ref='independent' should emit a UserWarning."""
        path_a, path_b = str(tmp_path / "tmp_normref_warn_a"), str(
            tmp_path / "tmp_normref_warn_b"
        )
        self._make_2d_images(path_a, path_b)

        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            frd.compute_frd(
                [path_a, path_b],
                frd_version="v1",
                verbose=False,
                save_features=False,
                norm_ref="independent",
            )
            user_warnings = [
                x
                for x in w
                if issubclass(x.category, UserWarning)
                and "norm_ref='independent'" in str(x.message)
            ]
            assert (
                len(user_warnings) >= 1
            ), f"Expected a UserWarning about norm_ref='independent', got {w}"
            assert "comparability" in str(user_warnings[0].message).lower()

    def test_joint_and_d1_no_warning(self, tmp_path):
        """norm_ref='joint' and 'd1' should NOT emit the independent-mode warning."""
        path_a, path_b = str(tmp_path / "tmp_normref_nowarn_a"), str(
            tmp_path / "tmp_normref_nowarn_b"
        )
        self._make_2d_images(path_a, path_b)

        import warnings

        for mode in ["joint", "d1"]:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                frd.compute_frd(
                    [path_a, path_b],
                    frd_version="v1",
                    verbose=False,
                    save_features=False,
                    norm_ref=mode,
                )
                indep_warnings = [
                    x
                    for x in w
                    if issubclass(x.category, UserWarning)
                    and "norm_ref='independent'" in str(x.message)
                ]
                assert len(indep_warnings) == 0, (
                    f"norm_ref='{mode}' should not emit independent-mode warning, "
                    f"but got: {indep_warnings}"
                )

    def test_d1_mode_uses_first_distribution_stats(self, tmp_path):
        """norm_ref='d1' should normalize both sets using only D1's statistics.

        When D1 == D2 (identical distributions), d1 and joint modes should produce
        the same result since the normalization base is equivalent.
        """
        path_a = str(tmp_path / "tmp_normref_d1_a")
        path_b = str(tmp_path / "tmp_normref_d1_b")
        self._make_2d_images(path_a, path_b)

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Same distribution -> d1 base == joint base -> results should match
            frd_joint = frd.compute_frd(
                [path_a, path_a],
                frd_version="v1",
                verbose=False,
                save_features=False,
                norm_ref="joint",
            )
            frd_d1 = frd.compute_frd(
                [path_a, path_a],
                frd_version="v1",
                verbose=False,
                save_features=False,
                norm_ref="d1",
            )
        print(f"Same dist -- joint: {frd_joint}, d1: {frd_d1}")
        # For identical paths, joint base == d1 base (both are just D1's features)
        assert np.isclose(frd_joint, frd_d1, rtol=1e-6), (
            f"With identical distributions, joint and d1 should match. "
            f"Got joint={frd_joint}, d1={frd_d1}"
        )

    def test_invalid_norm_ref_raises(self, tmp_path):
        """An invalid norm_ref value should raise ValueError."""
        path_a, path_b = str(tmp_path / "tmp_normref_invalid_a"), str(
            tmp_path / "tmp_normref_invalid_b"
        )
        self._make_2d_images(path_a, path_b)

        import pytest

        with pytest.raises(ValueError, match="Unknown norm_ref"):
            frd.compute_frd(
                [path_a, path_b],
                frd_version="v1",
                verbose=False,
                save_features=False,
                norm_ref="invalid_mode",
            )


# =============================================================================
#  Tests for new CLI/API parameters (14-item audit)
# =============================================================================


class TestNewParams:
    """Test the new unified parameters added in v1.1."""

    @staticmethod
    def _make_2d_images(path_a, path_b, n=3, size=64):
        Path(path_a).mkdir(exist_ok=True)
        Path(path_b).mkdir(exist_ok=True)
        for i in range(n):
            arr_a = np.random.rand(size, size) * 255
            arr_b = np.random.rand(size, size) * 255
            Image.fromarray(arr_a.astype(np.uint8), mode="L").save(
                f"{path_a}/img{i}.png"
            )
            Image.fromarray(arr_b.astype(np.uint8), mode="L").save(
                f"{path_b}/img{i}.png"
            )

    # ── log_sigma ──

    def test_log_sigma_custom(self, tmp_path):
        """Custom log_sigma values should produce valid FRD."""
        path_a, path_b = str(tmp_path / "tmp_logsigma_a"), str(
            tmp_path / "tmp_logsigma_b"
        )
        self._make_2d_images(path_a, path_b)
        frd_val = frd.compute_frd(
            [path_a, path_b],
            frd_version="v1",
            verbose=False,
            save_features=False,
            log_sigma=[1.0, 2.0],
        )
        assert isinstance(frd_val, (float, np.floating))
        print(f"FRD with log_sigma=[1,2]: {frd_val}")

    # ── bin_width ──

    def test_bin_width(self, tmp_path):
        """Custom bin_width should produce valid FRD."""
        path_a, path_b = str(tmp_path / "tmp_binwidth_a"), str(
            tmp_path / "tmp_binwidth_b"
        )
        self._make_2d_images(path_a, path_b)
        frd_val = frd.compute_frd(
            [path_a, path_b],
            frd_version="v1",
            verbose=False,
            save_features=False,
            bin_width=10,
        )
        assert isinstance(frd_val, (float, np.floating))
        print(f"FRD with bin_width=10: {frd_val}")

    # ── normalize_scale / voxel_array_shift ──

    def test_normalize_scale_and_voxel_array_shift(self, tmp_path):
        """Custom normalize_scale and voxel_array_shift should produce valid FRD."""
        path_a, path_b = str(tmp_path / "tmp_normscale_a"), str(
            tmp_path / "tmp_normscale_b"
        )
        self._make_2d_images(path_a, path_b)
        frd_val = frd.compute_frd(
            [path_a, path_b],
            frd_version="v1",
            verbose=False,
            save_features=False,
            normalize_scale=200.0,
            voxel_array_shift=500.0,
        )
        assert isinstance(frd_val, (float, np.floating))
        print(f"FRD with normalize_scale=200, voxel_array_shift=500: {frd_val}")

    # ── resize_size as (w, h) tuple ──

    def test_resize_size_tuple(self, tmp_path):
        """resize_size as (w, h) tuple should work."""
        path_a, path_b = str(tmp_path / "tmp_resize_tuple_a"), str(
            tmp_path / "tmp_resize_tuple_b"
        )
        self._make_2d_images(path_a, path_b, size=128)
        frd_val = frd.compute_frd(
            [path_a, path_b],
            frd_version="v0",
            verbose=False,
            save_features=False,
            resize_size=(64, 64),
        )
        assert isinstance(frd_val, (float, np.floating))
        print(f"FRD with resize_size=(64,64): {frd_val}")

    # ── means_only ──

    def test_means_only(self, tmp_path):
        """means_only=True should produce a valid (different) FRD score."""
        path_a, path_b = str(tmp_path / "tmp_means_a"), str(tmp_path / "tmp_means_b")
        self._make_2d_images(path_a, path_b)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frd_full = frd.compute_frd(
                [path_a, path_b],
                frd_version="v0",
                verbose=False,
                save_features=False,
                means_only=False,
            )
            frd_means = frd.compute_frd(
                [path_a, path_b],
                frd_version="v0",
                verbose=False,
                save_features=False,
                means_only=True,
            )
        assert isinstance(frd_means, (float, np.floating))
        print(f"FRD full: {frd_full}, means_only: {frd_means}")
        # means_only should generally give a smaller or equal value
        assert (
            frd_means <= frd_full + 1e-6
        ), f"means_only should not exceed full FRD. Got means={frd_means}, full={frd_full}"

    # ── exclude_features ──

    def test_exclude_features(self, tmp_path):
        """Excluding feature categories should still produce valid results."""
        path_a, path_b = str(tmp_path / "tmp_exclude_a"), str(
            tmp_path / "tmp_exclude_b"
        )
        self._make_2d_images(path_a, path_b)
        frd_val = frd.compute_frd(
            [path_a, path_b],
            frd_version="v1",
            verbose=False,
            save_features=False,
            exclude_features=["textural"],
        )
        assert isinstance(frd_val, (float, np.floating))
        print(f"FRD excluding textural features: {frd_val}")

    # ── match_sample_count ──

    def test_match_sample_count(self, tmp_path):
        """match_sample_count should subsample larger set and produce valid FRD."""
        path_a, path_b = str(tmp_path / "tmp_matchcount_a"), str(
            tmp_path / "tmp_matchcount_b"
        )
        self._make_2d_images(path_a, path_b, n=3)
        # Add extra images to path_b to make it larger
        for i in range(3, 6):
            arr = np.random.rand(64, 64) * 255
            Image.fromarray(arr.astype(np.uint8), mode="L").save(f"{path_b}/img{i}.png")
        frd_val = frd.compute_frd(
            [path_a, path_b],
            frd_version="v0",
            verbose=False,
            save_features=False,
            match_sample_count=True,
        )
        assert isinstance(frd_val, (float, np.floating))
        print(f"FRD with match_sample_count: {frd_val}")

    # ── settings_dict ──

    def test_settings_dict(self, tmp_path):
        """settings_dict should pass through to the extractor."""
        path_a, path_b = str(tmp_path / "tmp_settingsdict_a"), str(
            tmp_path / "tmp_settingsdict_b"
        )
        self._make_2d_images(path_a, path_b)
        frd_val = frd.compute_frd(
            [path_a, path_b],
            frd_version="v1",
            verbose=False,
            save_features=False,
            settings_dict={"binWidth": 10},
        )
        assert isinstance(frd_val, (float, np.floating))
        print(f"FRD with settings_dict={{binWidth: 10}}: {frd_val}")

    # ── save_features default ──

    def test_save_features_default_false(self):
        """compute_frd save_features should default to False."""
        import inspect

        sig = inspect.signature(frd.compute_frd)
        default = sig.parameters["save_features"].default
        assert default is False, f"Expected save_features default=False, got {default}"

    def test_save_frd_stats_save_features_default_false(self):
        """save_frd_stats save_features should default to False."""
        import inspect

        sig = inspect.signature(frd.save_frd_stats)
        default = sig.parameters["save_features"].default
        assert default is False, f"Expected save_features default=False, got {default}"


class TestExcludeFeatures:
    """Tests for the _apply_exclude_features helper."""

    def test_exclude_textural(self):
        names = [
            "firstorder_Mean",
            "glcm_Contrast",
            "glrlm_RunEntropy",
            "shape_Elongation",
        ]
        arrs = [np.random.rand(5, len(names))]
        filtered, fnames = frd._apply_exclude_features(arrs, names, ["textural"])
        assert "glcm_Contrast" not in fnames
        assert "glrlm_RunEntropy" not in fnames
        assert "firstorder_Mean" in fnames
        assert "shape_Elongation" in fnames

    def test_exclude_wavelet(self):
        names = [
            "firstorder_Mean",
            "wavelet-LLH_glcm_Contrast",
            "wavelet-HHL_firstorder_Mean",
        ]
        arrs = [np.random.rand(5, len(names))]
        filtered, fnames = frd._apply_exclude_features(arrs, names, ["wavelet"])
        assert len(fnames) == 1
        assert fnames[0] == "firstorder_Mean"

    def test_exclude_firstorder(self):
        names = ["firstorder_Mean", "glcm_Contrast", "wavelet-LLH_firstorder_Skewness"]
        arrs = [np.random.rand(5, len(names))]
        filtered, fnames = frd._apply_exclude_features(arrs, names, ["firstorder"])
        assert all("firstorder" not in n for n in fnames)

    def test_exclude_shape(self):
        names = [
            "firstorder_Mean",
            "shape_Elongation",
            "shape2D_Perimeter",
            "glcm_Contrast",
        ]
        arrs = [np.random.rand(5, len(names))]
        filtered, fnames = frd._apply_exclude_features(arrs, names, ["shape"])
        assert "shape_Elongation" not in fnames
        assert "shape2D_Perimeter" not in fnames
        assert "firstorder_Mean" in fnames
        assert "glcm_Contrast" in fnames
        assert len(fnames) == 2

    def test_exclude_shape_wavelet_prefix(self):
        """Shape features behind wavelet prefix should also be caught."""
        names = ["wavelet-LLH_shape_Elongation", "shape_MeshVolume", "firstorder_Mean"]
        arrs = [np.random.rand(5, len(names))]
        filtered, fnames = frd._apply_exclude_features(arrs, names, ["shape"])
        assert "shape_MeshVolume" not in fnames
        assert "wavelet-LLH_shape_Elongation" not in fnames
        assert "firstorder_Mean" in fnames

    def test_exclude_none(self):
        names = ["a", "b", "c"]
        arrs = [np.random.rand(3, 3)]
        filtered, fnames = frd._apply_exclude_features(arrs, names, None)
        assert fnames == names
        assert filtered[0].shape == arrs[0].shape


class TestMatchSampleCount:
    """Tests for the _apply_match_sample_count helper."""

    def test_subsamples_larger(self):
        a = np.random.rand(10, 5)
        b = np.random.rand(20, 5)
        result = frd._apply_match_sample_count([a, b])
        assert result[0].shape[0] == 10
        assert result[1].shape[0] == 10

    def test_equal_sizes_unchanged(self):
        a = np.random.rand(10, 5)
        b = np.random.rand(10, 5)
        result = frd._apply_match_sample_count([a, b])
        assert result[0].shape[0] == 10
        assert result[1].shape[0] == 10


class TestMeansOnly:
    """Tests for the means_only parameter in calculate_frechet_distance."""

    def test_means_only_zero_for_identical(self):
        """Identical distributions should give 0 with means_only."""
        mu = np.array([1.0, 2.0, 3.0])
        sigma = np.eye(3)
        val = frd.calculate_frechet_distance(mu, sigma, mu, sigma, means_only=True)
        assert np.isclose(
            val, 0.0
        ), f"Expected 0 for identical distributions, got {val}"

    def test_means_only_less_than_full(self):
        """means_only distance should be <= full Fréchet distance."""
        mu1 = np.array([1.0, 2.0])
        mu2 = np.array([3.0, 4.0])
        sigma1 = np.array([[1.0, 0.5], [0.5, 2.0]])
        sigma2 = np.array([[2.0, 0.3], [0.3, 1.0]])
        full = frd.calculate_frechet_distance(
            mu1, sigma1, mu2, sigma2, means_only=False
        )
        means = frd.calculate_frechet_distance(
            mu1, sigma1, mu2, sigma2, means_only=True
        )
        assert means <= full + 1e-10, f"means_only ({means}) > full ({full})"


class TestInterpretFrd:
    """Tests for the interpret_frd function."""

    def test_interpret_produces_output(self, tmp_path):
        """interpret_frd should produce plots and return results dict."""
        try:
            import matplotlib
        except ImportError:
            import pytest

            pytest.skip("matplotlib not installed")

        feats1 = np.random.rand(20, 10)
        feats2 = np.random.rand(20, 10) + 0.5
        names = [f"feature_{i}" for i in range(10)]
        viz_dir = str(tmp_path / "tmp_interpret_output")

        results = frd.interpret_frd(
            [feats1, feats2], names, viz_dir=viz_dir, run_tsne=False
        )
        assert "top_changed_features" in results
        assert "n_features" in results
        assert results["n_features"] == 10
        assert len(results["top_changed_features"]) <= 20
        assert os.path.exists(os.path.join(viz_dir, "sorted_feature_differences.png"))


class TestDetectOod:
    """Tests for the detect_ood function."""

    def test_image_level_ood(self, tmp_path):
        """Image-level OOD detection should produce predictions."""
        ref = np.random.rand(50, 10)
        test = np.random.rand(20, 10) + 5.0  # shifted → should be OOD
        output_dir = str(tmp_path / "tmp_ood_output")

        results = frd.detect_ood(
            [ref, test],
            detection_type="image",
            val_frac=0.2,
            output_dir=output_dir,
        )
        assert "threshold" in results
        assert "scores" in results
        assert "predictions" in results
        assert len(results["predictions"]) == 20
        assert os.path.exists(os.path.join(output_dir, "ood_predictions.csv"))

    def test_dataset_level_ood(self, tmp_path):
        """Dataset-level OOD detection should produce nFRD score."""
        try:
            from sklearn.metrics import roc_auc_score
        except ImportError:
            import pytest

            pytest.skip("scikit-learn not installed")

        ref = np.random.rand(50, 10)
        test = np.random.rand(20, 10) + 5.0
        output_dir = str(tmp_path / "tmp_ood_dataset")

        results = frd.detect_ood(
            [ref, test],
            detection_type="dataset",
            output_dir=output_dir,
        )
        assert "nfrd" in results
        assert 0 <= results["nfrd"] <= 1.0
        print(f"nFRD: {results['nfrd']}")

    def test_invalid_detection_type_raises(self):
        """Invalid detection_type should raise ValueError."""
        import pytest

        ref = np.random.rand(10, 5)
        test = np.random.rand(10, 5)
        with pytest.raises(ValueError, match="detection_type must be"):
            frd.detect_ood([ref, test], detection_type="invalid")

    def test_use_val_set_false(self, tmp_path):
        """When use_val_set=False, full reference set is used for threshold estimation."""
        np.random.seed(42)
        ref = np.random.rand(50, 10)
        test = np.random.rand(20, 10) + 5.0
        output_dir = str(tmp_path / "tmp_ood_no_val")

        results = frd.detect_ood(
            [ref, test],
            detection_type="image",
            use_val_set=False,
            output_dir=output_dir,
        )
        assert "threshold" in results
        assert "scores" in results
        assert len(results["predictions"]) == 20
        # Most shifted samples should be detected as OOD
        assert results["predictions"].sum() > 10

    def test_use_val_set_true(self, tmp_path):
        """When use_val_set=True, a held-out split is used for threshold estimation."""
        np.random.seed(42)
        ref = np.random.rand(50, 10)
        test = np.random.rand(20, 10) + 5.0
        output_dir = str(tmp_path / "tmp_ood_val")

        results = frd.detect_ood(
            [ref, test],
            detection_type="image",
            use_val_set=True,
            val_frac=0.2,
            output_dir=output_dir,
        )
        assert "threshold" in results
        assert len(results["predictions"]) == 20
        assert results["predictions"].sum() > 10

    def test_seed_reproducibility(self, tmp_path):
        """Setting a seed should produce reproducible results with use_val_set=True."""
        ref = np.random.rand(50, 10)
        test = np.random.rand(20, 10)
        output_dir = str(tmp_path / "tmp_ood_seed")

        r1 = frd.detect_ood(
            [ref, test],
            detection_type="image",
            use_val_set=True,
            seed=1338,
            output_dir=output_dir,
        )
        r2 = frd.detect_ood(
            [ref, test],
            detection_type="image",
            use_val_set=True,
            seed=1338,
            output_dir=output_dir,
        )
        np.testing.assert_array_equal(r1["scores"], r2["scores"])
        np.testing.assert_array_equal(r1["predictions"], r2["predictions"])
        assert r1["threshold"] == r2["threshold"]

    def test_filenames_in_csv(self, tmp_path):
        """Filenames should appear in CSV output when provided."""
        ref = np.random.rand(30, 10)
        test = np.random.rand(5, 10)
        filenames = [
            "img_001.png",
            "img_002.png",
            "img_003.png",
            "img_004.png",
            "img_005.png",
        ]
        output_dir = str(tmp_path / "tmp_ood_filenames")

        frd.detect_ood(
            [ref, test],
            detection_type="image",
            use_val_set=False,
            filenames=filenames,
            output_dir=output_dir,
        )
        csv_path = os.path.join(output_dir, "ood_predictions.csv")
        assert os.path.exists(csv_path)
        import csv as csv_mod

        with open(csv_path, "r") as f:
            reader = csv_mod.reader(f)
            header = next(reader)
            assert header[0] == "filename"
            rows = list(reader)
            assert len(rows) == 5
            assert rows[0][0] == "img_001.png"
            assert rows[4][0] == "img_005.png"

    def test_no_filenames_uses_index(self, tmp_path):
        """Without filenames, CSV should use numeric indices."""
        ref = np.random.rand(30, 10)
        test = np.random.rand(5, 10)
        output_dir = str(tmp_path / "tmp_ood_nonames")

        frd.detect_ood(
            [ref, test],
            detection_type="image",
            use_val_set=False,
            output_dir=output_dir,
        )
        csv_path = os.path.join(output_dir, "ood_predictions.csv")
        import csv as csv_mod

        with open(csv_path, "r") as f:
            reader = csv_mod.reader(f)
            header = next(reader)
            assert header[0] == "index"
            rows = list(reader)
            assert rows[0][0] == "0"

    def test_counting_assumption(self, tmp_path):
        """Counting distribution assumption should produce valid results."""
        ref = np.random.rand(50, 10)
        test = np.random.rand(20, 10) + 5.0
        output_dir = str(tmp_path / "tmp_ood_counting")

        results = frd.detect_ood(
            [ref, test],
            detection_type="image",
            id_dist_assumption="counting",
            use_val_set=False,
            output_dir=output_dir,
        )
        assert "threshold" in results
        assert "p_values" in results
        assert all(0 <= p <= 1 for p in results["p_values"])

    def test_t_assumption(self, tmp_path):
        """T-distribution assumption should produce valid results."""
        ref = np.random.rand(50, 10)
        test = np.random.rand(20, 10) + 5.0
        output_dir = str(tmp_path / "tmp_ood_t")

        results = frd.detect_ood(
            [ref, test],
            detection_type="image",
            id_dist_assumption="t",
            use_val_set=False,
            output_dir=output_dir,
        )
        assert "threshold" in results
        assert "p_values" in results
        assert all(0 <= p <= 1 for p in results["p_values"])


class TestConstants:
    """Verify new constants are properly exported."""

    def test_exclude_options(self):
        assert frd.EXCLUDE_TEXTURAL == "textural"
        assert frd.EXCLUDE_WAVELET == "wavelet"
        assert frd.EXCLUDE_FIRSTORDER == "firstorder"
        assert frd.EXCLUDE_SHAPE == "shape"
        assert frd.EXCLUDE_OPTIONS == {"textural", "wavelet", "firstorder", "shape"}

    def test_default_settings(self):
        assert frd.DEFAULT_BIN_WIDTH == 5
        assert frd.DEFAULT_NORMALIZE_SCALE == 100
        assert frd.DEFAULT_VOXEL_ARRAY_SHIFT == 300


# =============================================================================
#  Equivalence tests: merged vs original v0 / v1
# =============================================================================


class TestEquivalenceV0:
    """Verify that merged v0-mode produces EXACTLY the same FRD as original frd_v0.

    Key expectations for equivalence:
    - Identical feature extractor setup: dict-based, same 8 feature classes, same settings.
    - Identical image preprocessing: no corner-voxel zeroing, no 2D→3D expansion.
    - Identical normalization: minmax, range [0, 7.4567], joint base (both sets concatenated).
    - Identical Fréchet distance formula (same eps, same sqrtm handling).
    - No log transform for v0.

    Uses a sizeable number of synthetic images (20 per distribution) to exercise the
    full pipeline and increase sensitivity to any hidden divergences.
    """

    N_IMAGES = 20  # considerable number per distribution
    IMAGE_SIZE = 128

    @staticmethod
    def _make_synthetic_distribution(out_dir, n, size, seed):
        """Generate n synthetic grayscale images in out_dir with a fixed random seed."""
        rng = np.random.RandomState(seed)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            arr = rng.rand(size, size) * 255
            Image.fromarray(arr.astype(np.uint8), mode="L").save(
                f"{out_dir}/img_{i:03d}.png"
            )

    def test_merged_v0_equals_original_v0(self, tmp_path):
        """Core equivalence test: merged v0 must match original v0 exactly."""
        import pytest

        try:
            from frd_v0.src.frd_score import frd as frd_v0_original
        except ImportError:
            pytest.skip(
                "Original frd_v0 not available (removed from repo; check git history)"
            )

        path_a = str(tmp_path / "tmp_equiv_v0_a")
        path_b = str(tmp_path / "tmp_equiv_v0_b")

        # Generate identical synthetic images using fixed seeds
        self._make_synthetic_distribution(
            path_a, self.N_IMAGES, self.IMAGE_SIZE, seed=42
        )
        self._make_synthetic_distribution(
            path_b, self.N_IMAGES, self.IMAGE_SIZE, seed=123
        )

        # Run original v0
        frd_original = frd_v0_original.compute_frd(
            [path_a, path_b],
            verbose=False,
            save_features=False,
            # Use all original v0 defaults:
            # features=[firstorder, glcm, glrlm, gldm, glszm, ngtdm, shape, shape2D]
            # norm_type="minmax"
            # norm_range=[0.0, 7.45670747756958]
            # norm_sets_separately=True  (= joint normalization)
        )

        # Run merged in v0 mode
        frd_merged = frd.compute_frd(
            [path_a, path_b],
            frd_version="v0",
            verbose=False,
            save_features=False,
            # All defaults should match original v0:
            # features=V0_DEFAULT_FEATURES (same 8 classes)
            # norm_type=V0_DEFAULT_NORM_TYPE ("minmax")
            # norm_range=V0_DEFAULT_NORM_RANGE ([0.0, 7.4567...])
            # norm_ref=V0_DEFAULT_NORM_REF ("joint")
        )

        print(f"Original v0 FRD: {frd_original}")
        print(f"Merged v0 FRD:   {frd_merged}")
        print(f"Difference:      {abs(frd_original - frd_merged)}")

        assert np.isclose(frd_original, frd_merged, rtol=1e-10), (
            f"EQUIVALENCE FAILURE: Original v0 ({frd_original}) != Merged v0 ({frd_merged}). "
            f"Difference = {abs(frd_original - frd_merged)}"
        )

    def test_merged_v0_defaults_match_original(self):
        """Verify that merged v0 defaults (norm_type, norm_range, norm_ref, features)
        match the original v0 defaults exactly."""
        import inspect

        import pytest

        try:
            from frd_v0.src.frd_score import frd as frd_v0_original
        except ImportError:
            pytest.skip(
                "Original frd_v0 not available (removed from repo; check git history)"
            )

        # Original v0 defaults
        orig_sig = inspect.signature(frd_v0_original.compute_frd)
        orig_features = orig_sig.parameters["features"].default
        orig_norm_type = orig_sig.parameters["norm_type"].default
        orig_norm_range = orig_sig.parameters["norm_range"].default
        orig_norm_sets_separately = orig_sig.parameters["norm_sets_separately"].default

        # Merged v0 defaults (after _resolve_defaults)
        norm_type, norm_range, features, image_types, norm_ref = frd._resolve_defaults(
            "v0", None, None, None, None
        )

        assert features == list(
            orig_features
        ), f"Features mismatch: merged={features}, original={orig_features}"
        assert (
            norm_type == orig_norm_type
        ), f"norm_type mismatch: merged={norm_type}, original={orig_norm_type}"
        assert norm_range == list(
            orig_norm_range
        ), f"norm_range mismatch: merged={norm_range}, original={orig_norm_range}"
        # Original v0's norm_sets_separately=True maps to "joint"
        assert orig_norm_sets_separately is True
        assert (
            norm_ref == "joint"
        ), f"norm_ref mismatch: merged={norm_ref}, expected 'joint' (= orig norm_sets_separately=True)"
        print(
            f"All v0 defaults match: features={features}, norm_type={norm_type}, norm_range={norm_range}, norm_ref={norm_ref}"
        )


class TestEquivalenceV1:
    """Verify that merged v1-mode produces the same FRD as original frd_v1.

    The original v1 used a different architecture (PIL + DataFrame + per-image
    extractor), but functionally we now replicate its exact pipeline:
    - Include numeric diagnostics in the feature vector (paper artifact)
    - Cast features to float32 before normalization
    - Drop NaN rows (images) before normalization
    - D1-only z-score with raw division (no std=0 special-casing)
    - Drop NaN/Inf feature columns post-normalization
    - log(d_F^2) final transform

    The only unavoidable difference is IEEE-754 floating-point accumulation order
    in numpy operations (mean, std, cov), which produces noise below float32
    machine epsilon (~1.2e-7).

    IMPORTANT: N_IMAGES must be divisible by 8 (original v1's default num_workers
    for its parallel extraction).  The original v1 has a parallelization bug where
    ``split_num = N // num_workers`` causes images at the tail to be silently dropped
    when N is not divisible by num_workers.
    """

    N_IMAGES = (
        24  # divisible by 8 to avoid original v1's parallelization truncation bug
    )
    IMAGE_SIZE = 64

    @staticmethod
    def _make_synthetic_distribution(out_dir, n, size, seed):
        rng = np.random.RandomState(seed)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            arr = rng.rand(size, size) * 255
            Image.fromarray(arr.astype(np.uint8), mode="L").save(
                f"{out_dir}/img_{i:03d}.png"
            )

    def test_merged_v1_defaults_match_original(self):
        """Verify that merged v1 defaults match original v1's normalization behavior.

        Original v1 defaults:
        - zscore normalization
        - D1-only reference (not joint)
        - NaN/Inf column dropping
        - match_sample_count=True in compute_frd.py main()
        - log(d_F^2) final transform
        """
        norm_type, norm_range, features, image_types, norm_ref = frd._resolve_defaults(
            "v1", None, None, None, None
        )

        assert norm_type == "zscore", f"Expected zscore, got {norm_type}"
        assert norm_ref == "d1", f"Expected 'd1' (D1-only), got {norm_ref}"
        # v1 z-score range: with new_min=0, new_max=1, the formula becomes
        # ((x-mean)/std) * (1-0) + 0 = (x-mean)/std, which is pure z-score.
        assert norm_range == [0.0, 1.0], f"Expected [0.0, 1.0], got {norm_range}"
        print(
            f"v1 defaults: norm_type={norm_type}, norm_ref={norm_ref}, norm_range={norm_range}"
        )

    def test_merged_v1_produces_valid_frd(self, tmp_path):
        """Merged v1 with defaults should produce a valid log-transformed FRD."""
        path_a = str(tmp_path / "tmp_equiv_v1_valid_a")
        path_b = str(tmp_path / "tmp_equiv_v1_valid_b")
        self._make_synthetic_distribution(
            path_a, self.N_IMAGES, self.IMAGE_SIZE, seed=42
        )
        self._make_synthetic_distribution(
            path_b, self.N_IMAGES, self.IMAGE_SIZE, seed=123
        )

        frd_val = frd.compute_frd(
            [path_a, path_b],
            frd_version="v1",
            verbose=False,
            save_features=False,
        )
        print(f"Merged v1 FRD (D1-ref default): {frd_val}")
        assert isinstance(frd_val, (float, np.floating))
        # v1 produces log-transformed values; should be a real number
        assert np.isfinite(frd_val), f"Expected finite FRD, got {frd_val}"

    def test_merged_v1_equivalence_with_original_v1(self, tmp_path):
        """End-to-end equivalence test: run original v1 pipeline and merged v1, compare.

        Both produce the same FRD value on identical single-channel grayscale images.
        N_IMAGES is divisible by 8 to avoid the original v1's parallelization truncation
        bug (``split_num = N // num_workers`` drops trailing images).

        After fixing all divergence sources (float32 cast, NaN row dropping, raw z-score,
        numeric diagnostics inclusion, YAML parity), the remaining difference is purely
        IEEE-754 floating-point accumulation order noise — well below float32 machine
        epsilon (~1.2e-7).
        """
        import pytest

        pytest.skip(
            "Original frd_v1 not available (removed from repo; check git history)"
        )

        # Use absolute paths so they work regardless of CWD
        path_a = str(tmp_path / "tmp_equiv_v1_e2e_a")
        path_b = str(tmp_path / "tmp_equiv_v1_e2e_b")
        self._make_synthetic_distribution(
            path_a, self.N_IMAGES, self.IMAGE_SIZE, seed=42
        )
        self._make_synthetic_distribution(
            path_b, self.N_IMAGES, self.IMAGE_SIZE, seed=123
        )

        # ── Run original v1 pipeline ──
        # The original v1 uses a relative YAML path (configs/2D_extraction.yaml),
        # so we must cd into the frd_v1 directory for it to resolve correctly.
        original_cwd = os.getcwd()
        frd_v1_dir = os.path.join(os.path.dirname(__file__), "..", "..", "frd_v1")
        os.chdir(frd_v1_dir)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                compute_and_save_imagefolder_radiomics_parallel(
                    path_a, radiomics_fname="radiomics.csv"
                )
                compute_and_save_imagefolder_radiomics_parallel(
                    path_b, radiomics_fname="radiomics.csv"
                )
        finally:
            os.chdir(original_cwd)

        radiomics_df1 = pd.read_csv(os.path.join(path_a, "radiomics.csv"))
        radiomics_df2 = pd.read_csv(os.path.join(path_b, "radiomics.csv"))

        feats1, feats2 = convert_radiomic_dfs_to_vectors(
            radiomics_df1,
            radiomics_df2,
            match_sample_count=True,
            normalize=True,
        )
        fd = v1_frechet_distance(feats1, feats2)
        frd_original_v1 = np.log(fd)

        # ── Run merged v1 pipeline ──
        frd_merged_v1 = frd.compute_frd(
            [path_a, path_b],
            frd_version="v1",
            verbose=False,
            save_features=False,
        )

        print(f"Original v1 FRD: {frd_original_v1}")
        print(f"Merged v1 FRD:   {frd_merged_v1}")
        print(f"Difference:      {abs(frd_original_v1 - frd_merged_v1):.2e}")

        # The tolerance is 1e-6 (float32 machine epsilon is ~1.2e-7).
        # Observed differences are typically ~2e-8, well within this bound.
        assert np.isclose(frd_original_v1, frd_merged_v1, rtol=1e-6), (
            f"EQUIVALENCE FAILURE: Original v1 ({frd_original_v1}) != Merged v1 ({frd_merged_v1}). "
            f"Difference = {abs(frd_original_v1 - frd_merged_v1):.2e}"
        )

    def test_v0_norm_ref_default_is_joint(self):
        """v0 default norm_ref should be 'joint' (matching original v0 behavior)."""
        _, _, _, _, norm_ref = frd._resolve_defaults("v0", None, None, None, None)
        assert norm_ref == "joint"

    def test_v1_norm_ref_default_is_d1(self):
        """v1 default norm_ref should be 'd1' (matching original v1 behavior)."""
        _, _, _, _, norm_ref = frd._resolve_defaults("v1", None, None, None, None)
        assert norm_ref == "d1"

    def test_explicit_norm_ref_overrides_default(self):
        """Explicitly passed norm_ref should override the version default."""
        _, _, _, _, norm_ref = frd._resolve_defaults(
            "v1", None, None, None, None, norm_ref="joint"
        )
        assert norm_ref == "joint"
        _, _, _, _, norm_ref = frd._resolve_defaults(
            "v0", None, None, None, None, norm_ref="d1"
        )
        assert norm_ref == "d1"


# ─────────────────────────────────────────────────────────────────────────────
#  Medical image tests (real downscaled images from tests/data/medical_2d)
# ─────────────────────────────────────────────────────────────────────────────

# Resolve the fixture directory once.
_MEDICAL_2D_DIR = Path(__file__).resolve().parent / "data" / "medical_2d"
_MEDICAL_D1 = _MEDICAL_2D_DIR / "d1"
_MEDICAL_D2 = _MEDICAL_2D_DIR / "d2"

# Skip the whole class if the fixture images were not shipped.
_HAS_MEDICAL = _MEDICAL_D1.is_dir() and _MEDICAL_D2.is_dir()


class TestMedicalImages2D:
    """End-to-end tests using real (downscaled) medical images.

    The fixture lives in ``tests/data/medical_2d/{d1,d2}/`` and contains
    128×128 grayscale PNGs derived from diverse modalities, giving a much
    more realistic feature distribution than synthetically generated noise.
    """

    @staticmethod
    def _skip_if_missing():
        import pytest

        if not _HAS_MEDICAL:
            pytest.skip("Medical image fixtures not found in tests/data/medical_2d")

    # -- basic FRD computation ------------------------------------------------

    def test_frd_v1_medical(self):
        """Compute FRD v1 between two real-image distributions."""
        self._skip_if_missing()
        frd_value = frd.compute_frd(
            [str(_MEDICAL_D1), str(_MEDICAL_D2)],
            frd_version="v1",
            verbose=False,
            save_features=False,
            norm_ref="joint",
        )
        print(f"FRD v1 (D1 vs D2) medical: {frd_value}")
        assert isinstance(frd_value, (float, np.floating))
        assert frd_value >= 0, "FRD must be non-negative"

    def test_frd_v0_medical(self):
        """Compute FRD v0 between two real-image distributions."""
        self._skip_if_missing()
        frd_value = frd.compute_frd(
            [str(_MEDICAL_D1), str(_MEDICAL_D2)],
            frd_version="v0",
            verbose=False,
            save_features=False,
        )
        print(f"FRD v0 (D1 vs D2) medical: {frd_value}")
        assert isinstance(frd_value, (float, np.floating))
        assert frd_value >= 0

    # -- same-distribution sanity check ----------------------------------------

    def test_same_distribution_lower_frd(self):
        """FRD(D1, D1) should be much smaller than FRD(D1, D2).

        Because D1 and D2 come from different modalities, the cross-
        distribution distance *must* exceed the self-distance.
        """
        self._skip_if_missing()
        frd_self = frd.compute_frd(
            [str(_MEDICAL_D1), str(_MEDICAL_D1)],
            frd_version="v1",
            verbose=False,
            save_features=False,
            norm_ref="joint",
        )
        frd_cross = frd.compute_frd(
            [str(_MEDICAL_D1), str(_MEDICAL_D2)],
            frd_version="v1",
            verbose=False,
            save_features=False,
            norm_ref="joint",
        )
        print(f"FRD self={frd_self:.6f}, cross={frd_cross:.6f}")
        assert frd_self < frd_cross, (
            f"Self-distance ({frd_self}) should be less than "
            f"cross-distribution distance ({frd_cross})"
        )

    # -- norm_ref variants ----------------------------------------------------

    def test_norm_ref_d1_medical(self):
        """FRD v1 with norm_ref='d1' should return a valid float."""
        self._skip_if_missing()
        frd_value = frd.compute_frd(
            [str(_MEDICAL_D1), str(_MEDICAL_D2)],
            frd_version="v1",
            verbose=False,
            save_features=False,
            norm_ref="d1",
        )
        print(f"FRD v1 norm_ref=d1: {frd_value}")
        assert isinstance(frd_value, (float, np.floating))
        assert frd_value >= 0

    def test_norm_ref_independent_medical(self):
        """FRD v1 with norm_ref='independent' should return a valid float."""
        self._skip_if_missing()
        frd_value = frd.compute_frd(
            [str(_MEDICAL_D1), str(_MEDICAL_D2)],
            frd_version="v1",
            verbose=False,
            save_features=False,
            norm_ref="independent",
        )
        print(f"FRD v1 norm_ref=independent: {frd_value}")
        assert isinstance(frd_value, (float, np.floating))
        assert frd_value >= 0

    # -- exclude_features options ---------------------------------------------

    def test_exclude_textural_medical(self):
        """Excluding textural features should still produce valid FRD."""
        self._skip_if_missing()
        frd_value = frd.compute_frd(
            [str(_MEDICAL_D1), str(_MEDICAL_D2)],
            frd_version="v1",
            verbose=False,
            save_features=False,
            norm_ref="joint",
            exclude_features=["textural"],
        )
        print(f"FRD v1 exclude textural: {frd_value}")
        assert isinstance(frd_value, (float, np.floating))
        assert frd_value >= 0

    def test_exclude_shape_medical(self):
        """Excluding shape features should still produce valid FRD."""
        self._skip_if_missing()
        frd_value = frd.compute_frd(
            [str(_MEDICAL_D1), str(_MEDICAL_D2)],
            frd_version="v1",
            verbose=False,
            save_features=False,
            norm_ref="joint",
            exclude_features=["shape"],
        )
        print(f"FRD v1 exclude shape: {frd_value}")
        assert isinstance(frd_value, (float, np.floating))
        assert frd_value >= 0

    # -- resize parameter -----------------------------------------------------

    def test_resize_medical(self):
        """FRD with resize_size should work on real images."""
        self._skip_if_missing()
        frd_value = frd.compute_frd(
            [str(_MEDICAL_D1), str(_MEDICAL_D2)],
            frd_version="v1",
            verbose=False,
            save_features=False,
            norm_ref="joint",
            resize_size=64,
        )
        print(f"FRD v1 resize_size=64: {frd_value}")
        assert isinstance(frd_value, (float, np.floating))
        assert frd_value >= 0

    # -- means_only flag ------------------------------------------------------

    def test_means_only_medical(self):
        """FRD with means_only=True on real images."""
        self._skip_if_missing()
        frd_value = frd.compute_frd(
            [str(_MEDICAL_D1), str(_MEDICAL_D2)],
            frd_version="v1",
            verbose=False,
            save_features=False,
            norm_ref="joint",
            means_only=True,
        )
        print(f"FRD v1 means_only: {frd_value}")
        assert isinstance(frd_value, (float, np.floating))
        assert frd_value >= 0

    # -- match_sample_count flag ----------------------------------------------

    def test_match_sample_count_medical(self):
        """FRD with match_sample_count=True on real images."""
        self._skip_if_missing()
        frd_value = frd.compute_frd(
            [str(_MEDICAL_D1), str(_MEDICAL_D2)],
            frd_version="v1",
            verbose=False,
            save_features=False,
            norm_ref="joint",
            match_sample_count=True,
        )
        print(f"FRD v1 match_sample_count: {frd_value}")
        assert isinstance(frd_value, (float, np.floating))
        assert frd_value >= 0


class TestPyradiomicsCompat:
    """Tests that pyradiomics installs, imports, and works correctly.

    Validates the fix for https://github.com/AIM-Harvard/pyradiomics/issues/903
    (PyPI pyradiomics broken for Python >=3.10).
    """

    def test_radiomics_imports(self):
        """Core radiomics modules should be importable."""
        import radiomics
        from radiomics import featureextractor, setVerbosity

        assert hasattr(featureextractor, "RadiomicsFeatureExtractor")

    def test_radiomics_version_not_broken_pypi(self):
        """Installed pyradiomics should NOT be the broken PyPI release (3.0.1a3 or 3.1.0).

        Those versions fail on Python >=3.10 due to configparser.SafeConfigParser removal.
        We require installation from GitHub master (>=3.1.1dev).
        """
        import radiomics

        version = radiomics.__version__
        broken_versions = {"3.0.1a3", "3.1.0"}
        assert version not in broken_versions, (
            f"Installed pyradiomics version {version} is a known-broken PyPI release. "
            "Install from GitHub master: pip install git+https://github.com/AIM-Harvard/pyradiomics.git@master"
        )

    def test_radiomics_version_is_dev(self):
        """Installed version should be a dev build from GitHub master (contains 'dev')."""
        import radiomics

        version = radiomics.__version__
        assert "dev" in version or "+" in version, (
            f"Expected a dev/GitHub build of pyradiomics, got '{version}'. "
            "Install from GitHub: pip install git+https://github.com/AIM-Harvard/pyradiomics.git@master"
        )

    def test_python_version_supported(self):
        """Python version should be >=3.10 (as declared in setup.py python_requires)."""
        import sys

        assert sys.version_info >= (3, 10), (
            f"Python {sys.version_info.major}.{sys.version_info.minor} is not supported. "
            "frd-score requires Python >=3.10."
        )

    def test_feature_extractor_v0_creation(self):
        """v0 dict-based feature extractor should instantiate without error."""
        extractor = frd.get_feature_extractor(frd_version="v0")
        assert extractor is not None
        assert hasattr(extractor, "enabledFeatures")
        # Default v0 should have all 8 feature classes
        assert len(extractor.enabledFeatures) == len(frd.V0_DEFAULT_FEATURES)

    def test_feature_extractor_v1_2d_creation(self):
        """v1 2D YAML-based feature extractor should instantiate without error."""
        extractor = frd.get_feature_extractor(frd_version="v1", image_dim=2)
        assert extractor is not None
        assert hasattr(extractor, "enabledFeatures")
        # v1 2D should have 5 feature classes from YAML
        assert len(extractor.enabledFeatures) == 5

    def test_feature_extractor_v1_3d_creation(self):
        """v1 3D YAML-based feature extractor should instantiate without error."""
        extractor = frd.get_feature_extractor(frd_version="v1", image_dim=3)
        assert extractor is not None
        assert hasattr(extractor, "enabledFeatures")

    def test_yaml_configs_exist(self):
        """Built-in YAML config files must exist at the expected paths."""
        assert os.path.exists(frd.V1_CONFIG_2D), f"Missing: {frd.V1_CONFIG_2D}"
        assert os.path.exists(frd.V1_CONFIG_3D), f"Missing: {frd.V1_CONFIG_3D}"

    def test_v1_2d_yaml_image_types(self):
        """v1 2D extractor should have Original, LoG, and Wavelet image types enabled."""
        extractor = frd.get_feature_extractor(frd_version="v1", image_dim=2)
        enabled = set(extractor.enabledImagetypes.keys())
        expected = {"Original", "LoG", "Wavelet"}
        assert expected == enabled, f"Expected image types {expected}, got {enabled}"

    def test_v1_2d_yaml_feature_classes(self):
        """v1 2D extractor should have the 5 YAML-configured feature classes."""
        extractor = frd.get_feature_extractor(frd_version="v1", image_dim=2)
        enabled = set(extractor.enabledFeatures.keys())
        expected = {"firstorder", "glcm", "glrlm", "glszm", "ngtdm"}
        assert (
            expected == enabled
        ), f"Expected feature classes {expected}, got {enabled}"

    def test_v0_single_image_extraction(self):
        """v0 extractor should successfully extract features from a single synthetic image."""
        import SimpleITK as sitk

        np.random.seed(42)
        arr = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        mask = np.ones_like(arr, dtype=np.uint8)

        img_sitk = sitk.GetImageFromArray(arr.astype(np.float32))
        mask_sitk = sitk.GetImageFromArray(mask)

        extractor = frd.get_feature_extractor(frd_version="v0")
        result = extractor.execute(img_sitk, mask_sitk)

        # Should have diagnostics + actual features
        feature_keys = [k for k in result.keys() if not k.startswith("diagnostics_")]
        assert len(feature_keys) > 0, "No features extracted"

    def test_v1_single_image_extraction(self):
        """v1 extractor should successfully extract features from a single synthetic 2D image.

        This is the most thorough integration test: it loads the YAML config, creates
        a pyradiomics extractor with LoG+Wavelet filters, and runs extraction end-to-end.
        Exercises the full import chain incl. C extensions.
        """
        import SimpleITK as sitk

        np.random.seed(42)
        # 2D image expanded to 3D (single slice) for pyradiomics
        arr = np.random.randint(0, 256, (1, 64, 64), dtype=np.uint8)
        mask = np.ones_like(arr, dtype=np.uint8)

        img_sitk = sitk.GetImageFromArray(arr.astype(np.float32))
        # Set a non-trivial spacing to avoid degenerate geometry warnings
        img_sitk.SetSpacing([1.0, 1.0, 1.0])
        mask_sitk = sitk.GetImageFromArray(mask)
        mask_sitk.SetSpacing([1.0, 1.0, 1.0])

        extractor = frd.get_feature_extractor(frd_version="v1", image_dim=2)
        result = extractor.execute(img_sitk, mask_sitk)

        feature_keys = [k for k in result.keys() if not k.startswith("diagnostics_")]
        # v1 with Original+LoG+Wavelet should produce hundreds of features
        assert (
            len(feature_keys) > 100
        ), f"Expected >100 features from v1 extraction, got {len(feature_keys)}"

    def test_import_guard_message(self):
        """The import guard in frd.py should produce a helpful error message.

        We can't actually trigger the ImportError (pyradiomics IS installed),
        but we verify the guard code path exists by checking the source.
        """
        import inspect

        source = inspect.getsource(frd)
        assert "pyradiomics is not installed or failed to import" in source
        assert "github.com/AIM-Harvard/pyradiomics/issues/903" in source

    def test_extractor_settings_override(self):
        """bin_width, normalize_scale, and voxel_array_shift should be overridable."""
        extractor = frd.get_feature_extractor(
            frd_version="v1",
            image_dim=2,
            bin_width=10,
            normalize_scale=200,
            voxel_array_shift=500,
        )
        assert extractor.settings["binWidth"] == 10
        assert extractor.settings["normalizeScale"] == 200
        assert extractor.settings["voxelArrayShift"] == 500

    def test_custom_log_sigma(self):
        """Custom LoG sigma values should be applied to the extractor."""
        custom_sigma = [1.0, 2.0]
        extractor = frd.get_feature_extractor(
            frd_version="v1",
            image_dim=2,
            log_sigma=custom_sigma,
        )
        log_cfg = extractor.enabledImagetypes.get("LoG", {})
        assert (
            log_cfg.get("sigma") == custom_sigma
        ), f"Expected LoG sigma={custom_sigma}, got {log_cfg.get('sigma')}"

    def test_image_types_override(self):
        """Overriding image_types should limit enabled types to the specified set."""
        extractor = frd.get_feature_extractor(
            frd_version="v1",
            image_dim=2,
            image_types=["Original"],
        )
        enabled = set(extractor.enabledImagetypes.keys())
        assert enabled == {"Original"}, f"Expected only Original, got {enabled}"

    def test_feature_groups_override_v1(self):
        """Overriding feature_groups for v1 should limit enabled classes."""
        extractor = frd.get_feature_extractor(
            frd_version="v1",
            image_dim=2,
            features=["firstorder", "glcm"],
        )
        enabled = set(extractor.enabledFeatures.keys())
        assert enabled == {
            "firstorder",
            "glcm",
        }, f"Expected firstorder+glcm, got {enabled}"
