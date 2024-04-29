"""Calculates the Frechet Radiomics Distance (FRD) to compare image biomarker distributions

The FRD metric calculates the distance between two distributions of biomarkers extracted from imaging data.
Typically, one of the two distributions is generated by a generative feature_extractor such as GANs or Diffusion feature_extractors.
When run as a stand-alone program, it compares the distribution of images that are stored as PNG/JPEG/NIfTI at
a specified location.

Some code from https://github.com/mseitzer/pytorch-fid and https://github.com/bioinf-jku/TTUR was reused and adapted.

Usage:
    python frd.py dir1 dir2

"""

import argparse
import csv
import logging
import os
import pathlib
import time
from pathlib import Path

import cv2
import numpy as np
from scipy import linalg
from tqdm import tqdm
from radiomics import featureextractor
import SimpleITK as sitk

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

# correctMask: Resize mask if there is a size mismatch between image and mask
# minimumROIDimensions: Set the minimum number of dimensions for a ROI mask. Needed to avoid error, as in our datasets we may have some mask_lists with dim=1.
# https://pyradiomics.readthedocs.io/en/latest/radiomics.html#radiomics.imageoperations.checkMask
# force2D: True is needed to extract 2d shape features when 'shape2d' is passed as feature name alongside 3d data
# Future work: Allow users to more easily access and adjust the settings dictionary (e.g. via command line arguments)
RADIOMICS_SETTINGS = {
    "correctMask": True,
    "minimumROIDimensions": 1,
}  # "force2D": True,}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculates the Frechet Radiomics Distance between two distributions of extracted pyradiomics "
        "features."
    )
    parser.add_argument(
        "paths",
        type=str,
        nargs=2,
        help="The two paths to the generated images or to .npz statistic file_lists",
    )

    parser.add_argument(
        "-m",
        "--paths_masks",
        type=str,
        nargs=2,
        default=None,
        help="The two paths to the folder where the mask file_lists are located.",
    )


    parser.add_argument(
        "-f",
        "--feature_groups",
        nargs='+',
        type=str,
        default=[
            "firstorder",
            "glcm",
            "glrlm",
            "gldm",
            "glszm",
            "ngtdm",
            "shape",
            "shape2D",
        ],
        help="The pyradiomics feature groups to be used for the frd calculation. Can be 'firstorder', "
        "'glcm', 'glrlm', 'gldm', 'glszm', 'ngtdm', 'shape', 'shape2D' ",
    )

    parser.add_argument(
        "-r",
        "--resize_size",
        type=int,
        default=None,
        help="In case the input images (and mask_lists) are to be resized to a specific pixel dimension. "
    )

    parser.add_argument(
        "-s",
        "--save_stats",
        action="store_true",
        help="Generate an npz archive from a directory of samples. The first paths is used as input and "
        "the second as output.",
    )

    parser.add_argument(
        "-R",
        "--norm_range",
        nargs=2,
        type=float,
        default=[0., 7.45670747756958],
        help="The allowed value range of features. Based on these values the frd features will be "
        "normalized. The range should be [min, max]. Default is [0, 7.45670747756958]. "
        "If norm_type is 'zscore', we recommend ignoring normalization range by setting "
        "it to [0, 1].",
    )

    parser.add_argument(
        "-T",
        "--norm_type",
        type=str,
        default="minmax",
        help="The strategy with which the frd features will be normalized. Can be 'minmax' "
        "or 'zscore'.",
    )

    parser.add_argument(
        "-A",
        "--norm_across",
        action="store_true",
        help="If true, the normalization (e.g., minmax or zscore) as well as rescaling to norm_range "
             "will be done based on all features from both datasets (e.g. syn, real) instead of on the features from "
             "each dataset separately.",
    )

    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="you may use this option to provide a csv file (e.g., paths/to/feature_names.csv) with the feature names to be used for the "
        "frd calculation. The csv file should have a single column with the feature names.",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="You may enable more detailed logging (logging.info) console logs by providing the 'verbose' argument.",
    )

    parser.add_argument(
        "-F",
        "--save_features",
        action="store_true",
        help="Indicates whether radiomics feature values (normalized and non-normalized) should be stored in a csv file. "
        "This can be useful for reproducibility and interpretability.",
    )

    args = parser.parse_args()
    return args


def compute_features(
    files,
    feature_extractor,
    masks=None,
    resize_size=None,
    verbose=False,
):
    """Calculates the features of the given query image (optionally, alongside a respective segmentation mask).

    Params:
    -- file_lists       : List of image file_lists paths
    -- feature_extractor       : Instance of radiomics feature_extractor
    -- mask_lists : The list of paths of the mask file_lists
    -- resize_size: In case the images should be resized before the radiomics features are calculated
    -- verbose: Indicates the verbosity level of the logging. If true, more info is logged to console.

    Returns:
    -- A numpy array of dimension (num images, num_features) that contains the
       extracted features of the given query image (optionally, alongside a respective segmentation mask).
    """
    prediction_array = None
    image_paths = []
    mask_paths = []
    radiomics_results = []

    for idx, file_path in enumerate(files):
        image_paths.append(file_path)
        if masks is not None:
            # Note: Sorting assumption here: Masks and images are in separate folders. Each image has a mask and
            # mask and image file are named similarly enough that sorting assures correspondence between image and mask index positions.
            try:
                mask_paths.append(masks[idx])
            except IndexError as e:
                raise RuntimeError(
                    f"Mask '{idx}' not found for image file '{file_path}'. "
                    f"Please revise that for each image there is a mask present. "
                    f"Ensure file names of image and mask correspond, as index position "
                    f"correspondence is assumed after sorting both image and mask lists. "
                    f"Exception: {e}"
                )
        else:
            mask_paths.append(None)

    total = len(image_paths)

    with tqdm(total=total) as pbar:
        for i, (image_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
            sitk_image = sitk.ReadImage(str(image_path), outputPixelType=sitk.sitkFloat32)
            if mask_path is None:
                # https://discourse.slicer.org/t/features-extraction/11047/3
                ma_arr = np.ones(sitk_image.GetSize()[::-1]).astype(
                    np.uint8
                )  # reverse the order as image is xyz, array is zyx
                sitk_mask = sitk.GetImageFromArray(ma_arr)
                try:
                    sitk_mask.CopyInformation(sitk_image)  # Copy geometric info
                except Exception as e:
                    logging.debug(
                        f"Error while trying to copy information from image to mask: {e}"
                    )
                    pass
                if verbose:
                    logging.debug(
                        "Empty mask (true everywhere) is used for feature extraction."
                    )
            else:
                sitk_mask = sitk.ReadImage(str(mask_path))

            # Check if the mask is in range [0, 255] and rescale it to [0, 1]
            if np.max(sitk.GetArrayViewFromImage(sitk_mask)) == 255:
                sitk_mask = sitk.Cast(sitk_mask, sitk.sitkFloat32) / 255.0

            if verbose and i % 100 == 0:
                # get some logging.infos to check the progress and if everything is working
                logging.info(
                    f"Now processing corresponding image-mask pair (IMG:{image_path}, MASK: {mask_path}. Do these correspond?"
                )

            if resize_size is not None:
                sitk_image_array = sitk.GetArrayViewFromImage(sitk_image)
                sitk_image_array_resized = resize_image_array(
                    sitk_image_array, resize_size, interpolation=cv2.INTER_LINEAR
                )
                sitk_image_resized = sitk.GetImageFromArray(sitk_image_array_resized)
                try:
                    sitk_image_resized.CopyInformation(sitk_image)
                except:
                    pass
                sitk_image = sitk_image_resized  # Update the image to the resized version

                sitk_mask_array = sitk.GetArrayViewFromImage(sitk_mask)
                sitk_mask_array_resized = resize_image_array(
                    sitk_mask_array, resize_size, interpolation=cv2.INTER_LINEAR
                )
                # After resizing, set all values above 0.5 to 1 and all values below to 0
                sitk_mask_array_resized[sitk_mask_array_resized > 0.5] = 1
                sitk_mask_array_resized[sitk_mask_array_resized <= 0.5] = 0
                sitk_mask_resized = sitk.GetImageFromArray(sitk_mask_array_resized)
                try:
                    sitk_mask_resized.CopyInformation(sitk_mask)
                except:
                    pass
                sitk_mask = sitk_mask_resized

            # Check if the mask contains only one voxel. This needs to be done before and after resizing as the mask
            if np.sum(sitk.GetArrayViewFromImage(sitk_mask)) <= 1:
                if verbose:
                    logging.info(
                        f"Skipping mask (after potentially having applied resizing to {resize_size}) with only one segmented voxel:",
                        mask_path,
                    )
                continue

            # Finally, run the feature extraction
            try:
                output = feature_extractor.execute(sitk_image, sitk_mask)
            except Exception as e:
                logging.debug(f"sitk_mask: {(sitk.GetArrayViewFromImage(sitk_mask))}")
                logging.debug(f"sitk_image: {(sitk.GetArrayViewFromImage(sitk_image))}")
                logging.debug(
                    f"shape sitk_mask: {(sitk.GetArrayViewFromImage(sitk_mask)).shape} and shape sitk_image: {(sitk.GetArrayViewFromImage(sitk_image)).shape}"
                )
                logging.error(
                    f"Error occurred while extracting features for image {i} from image {image_path} and mask {mask_path}: {e}"
                )
                raise e
            radiomics_features = {}
            for feature_name in output.keys():
                if "diagnostics" not in feature_name:
                    radiomics_features[feature_name.replace("original_", "")] = float(
                        output[feature_name]
                    )
            radiomics_results.append(radiomics_features)
            if verbose:
                logging.debug(
                    f"img_shape:{sitk.GetArrayViewFromImage(sitk_image).shape}, features: {len(list(radiomics_features.values()))}"
                )

            try:
                # We check if pred_arr is defined already.
                pred_arr
            except NameError:
                # As pred_arr was not yet defined we initialize it here based on the length/dimensionality of radiomics features
                pred_arr = np.empty(
                    (len(image_paths), len(list(radiomics_features.values())))
                )

            pred_arr[i] = list(radiomics_features.values())
            if verbose:logging.debug(f"Total number of features extracted for image {i}: {len(pred_arr[i])}")
            pbar.update(1)

    if radiomics_results and verbose:
        logging.info(f"Number of radiomics features: {len(radiomics_results[0])}")
    try:
        prediction_array = pred_arr
    except NameError:
        pass

    return prediction_array, radiomics_results, image_paths, mask_paths


def resize_image_array(sitk_image_array, resize_size, interpolation=cv2.INTER_LINEAR):
    if len(sitk_image_array.shape) == 2:
        sitk_image_array_resized = cv2.resize(
            sitk_image_array, (resize_size, resize_size), interpolation=interpolation
        )
    elif len(sitk_image_array.shape) == 3:
        # Going through z axis (which should be at index position 0 here in sitk image, and resizing each slice
        sitk_image_array_resized = np.zeros(
            (sitk_image_array.shape[0], resize_size, resize_size)
        )
        for j in range(sitk_image_array.shape[0]):
            sitk_image_array_resized[j] = cv2.resize(
                sitk_image_array[j],
                (resize_size, resize_size),
                interpolation=interpolation,
            )
    else:
        raise ValueError(
            f"SITK Image array has an unexpected shape: {sitk_image_array.shape}. Expected 2D or 3D array (no channel dim). Got {len(sitk_image_array.shape)}"
        )
    return sitk_image_array_resized


def save_features_to_csv(csv_file_path, image_paths, mask_paths, feature_data):
    """Save the feature data to a CSV file.

    Params:
    -- csv_file_path   : Path to the CSV file where the results will be saved
    -- image_paths     : List of image file paths. NOTE: Normally, the parent folder of this folder is where the csv features will be saved
    -- mask_paths      : List of mask file paths
    -- feature_data    : Feature data to be saved in the CSV file
    """

    with open(csv_file_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)

        # Write the header row
        header = ["image_path", "mask_path"]
        for feature_name in feature_data[0].keys():
            header.append(feature_name)
        writer.writerow(header)

        # Write the rows for each image
        for image_path, mask_path, features in zip(
            image_paths, mask_paths, feature_data
        ):
            # if mask_path is not None:
            #    mask_path = mask_path.with_name(mask_path.name.replace("_img_synth.jpg", "_mask_synth.jpg"))
            row = [str(image_path), str(mask_path)]
            row.extend(features.values())
            writer.writerow(row)

        # Compute and save the min and max values for each column
        num_features = len(feature_data[0])
        min_values = [
            np.min([data[feature_name] for data in feature_data])
            for feature_name in feature_data[0].keys()
        ]
        max_values = [
            np.max([data[feature_name] for data in feature_data])
            for feature_name in feature_data[0].keys()
        ]
        empty_row = [""] * (
            num_features + 2
        )  # Create an empty row to separate the data

        # Write the rows for min values
        writer.writerow(empty_row)
        writer.writerow(["Min", ""] + min_values)

        # Write the rows for max values
        writer.writerow(empty_row)
        writer.writerow(["Max", ""] + max_values)
    logging.info(f"Feature data saved to {csv_file_path}.")


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the features of the first set of samples.
    -- mu2   : Numpy array containing the features of the second set of samples.
    -- sigma1: The covariance matrix over features for the first set of samples.
    -- sigma2: The covariance matrix over features for the second set of samples.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "frechet distance calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        logging.debug(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            # raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def z_score_normalize(
    features, new_min=0, new_max=1, replace_nan=True, strict=False, feature_names=None, base_distribution=None,
):
    """Calculate the z score normalisation values of each feature across all images"""

    mean_values = np.nanmean(features, axis=0) if base_distribution is None else np.nanmean(base_distribution, axis=0)
    std_values = np.nanstd(features, axis=0) if base_distribution is None else np.nanstd(base_distribution, axis=0)

    # Create a new copy of features to perform normalization
    normalized_features = np.copy(features)

    # Perform z-score normalization for columns with different mean and std values
    for idx, (mean_val, std_val) in enumerate(zip(mean_values, std_values)):
        if not np.isnan(mean_val) and not np.isnan(std_val):
            feature_name = (
                feature_names[idx] if feature_names is not None else f"feature_{idx}"
            )
            # first we calculate the z-score, and then we normalize it to the new range
            # Note: We decided against shifting the distribution mean and std like in https://math.stackexchange.com/a/2908415
            # The reason for this is that we do not know the target mean, as we only have the target min and max
            # Also, we prefer not to change the std to avoid losing information and artificially removing/adding variation to the final FRD score
            # For this reason we stick to min-max normalization based on the z-score standardised values.
            # The downside is that the z-score values are not in the range between 0 and 1, so resulting values can be
            # lower than new_min and larger than new_max. This can increase the variation thereby affecting the FRD score.
            # For this reason we recommend (as a default), to use minmax instead of z-score normalization in FRD.
            logging.debug(
                f"{feature_name} std_val: {std_val}, mean_val: {mean_val}"
            )  # watch out as std can be very low or 0
            if std_val == 0 and strict:
                raise ValueError(
                    f"Warning: While calculating z-score (idx={idx}), a standard deviation of 0 was detected for feature {feature_name} (mean: {mean_val}). Please check the data for constant values. "
                    f"You may set norm_type to 'minmax' to avoid this issue."
                )
            if std_val == 0:
                logging.warning(
                    f"Warning: While calculating z-score (idx={idx}), a standard deviation of 0 was detected for feature {feature_name} (mean: {mean_val}). Fallback: Now replacing these feature values with 0 + new_min. new_min={new_min}. "
                    f"Alternatively, you may run again by setting norm_type to 'minmax' to avoid this issue."
                )
                normalized_features[:, idx] = 0 + float(new_min)
            else:
                normalized_features[:, idx] = (
                    (features[:, idx] - mean_val) / std_val
                ) * (float(new_max) - float(new_min)) + float(new_min)
            logging.debug(
                f"initial features: {features[:, idx]}. Their z-scores: {normalized_features[:, idx]}. Feature names: {feature_names}."
            )

    if replace_nan:
        # Replace NaN values with the mean of new_min and new_max
        nan_indices = np.isnan(normalized_features)
        mean_value = (float(new_min) + float(new_max)) / 2
        normalized_features[nan_indices] = mean_value

    return normalized_features


def min_max_normalize(features, new_min, new_max, replace_nan=True, feature_names=None, base_distribution=None):
    """Calculate the minimum and maximum values of each feature across all images"""

    min_values = np.nanmin(features, axis=0) if base_distribution is None else np.nanmin(base_distribution, axis=0)
    max_values = np.nanmax(features, axis=0) if base_distribution is None else np.nanmax(base_distribution, axis=0)

    # Create a new copy of features to perform normalization
    normalized_features = np.copy(features)

    # Perform Min-Max normalization for columns with different min and max values
    for idx, (min_val, max_val) in enumerate(zip(min_values, max_values)):
        if not np.isnan(min_val) and not np.isnan(max_val):
            if (max_val - min_val) == 0:
                feature_name = (
                    feature_names[idx]
                    if feature_names is not None
                    else f"feature_{idx}"
                )
                logging.warning(
                    f"Warning: While calculating minmax value (idx={idx}), a max_val - min_val ({max_val}-{min_val}) "
                    f"resulted in 0 for feature {feature_name} . Fallback: Now replacing feature value with 0.5 "
                    f"before scaling to new range ({float(new_min)}, {float(new_max)})."
                )
                normalized_features[:, idx] = 0.5 * (
                    float(new_max) - float(new_min)
                ) + float(new_min)
            else:
                normalized_features[:, idx] = (
                    (features[:, idx] - min_val) / (max_val - min_val)
                ) * (float(new_max) - float(new_min)) + float(new_min)

    if replace_nan:
        # Replace NaN values with the mean of new_min and new_max
        nan_indices = np.isnan(normalized_features)
        mean_value = (new_min + new_max) / 2
        normalized_features[nan_indices] = mean_value

    logging.debug(
        f"initial features: {features[:, idx]}. minmax standardized features: {normalized_features[:, idx]}. Feature names: {feature_names}."
    )

    return normalized_features


def calculate_feature_statistics(
    file_lists: list,
    norm_type: str,
    norm_range: list,
    feature_extractor,
    mask_lists: list=[None, None],
    resize_size: int=None,
    verbose: bool=False,
    save_features: bool=False,
    norm_sets_separately: bool=True,
) -> (list,list):
    """Calculation of the statistics used by the FRD.
    Params:
    -- file_lists                : List of image file_lists paths
    -- norm_type : The method with which the extracted features should be normalized
    -- norm_range : The range of normalization to scale the extracted features to after normalization
    -- feature_extractor    : Instance of pyradiomics feature_extractor
    -- mask_lists : The list of paths of the mask file_lists
    -- resize_size: In case the images should be resized before the radiomics features are calculated
    -- verbose: Indicates the verbosity level of the logging. If true, more info is logged to console.

    Returns:
    -- mu    : The mean over features extracted by the pyradiomics feature_extractor.
    -- sigma : The covariance matrix of the features extracted by the pyradiomics feature_extractor.
    """

    feature_list = []

    for idx, file_list in enumerate(file_lists):
        features, radiomics_results, image_paths, mask_paths = compute_features(
            files=file_list,
            feature_extractor=feature_extractor,
            masks=mask_lists[idx] if mask_lists is not None else None,
            resize_size=resize_size,
            verbose=verbose,
        )
        if verbose:
            logging.debug(f"features of radiomics: {features}")
            logging.debug(f"features of radiomics shape: {type(features)}")

        # to check NaN values in features
        if np.isnan(features).any():
            nan_indices = np.where(np.isnan(features))
            unique_nan_indices = np.unique(nan_indices[1])
            logging.warning("Warning: NaN values detected in the features array.")
            if verbose:
                logging.info("Number of NaN values for each feature:")
            for feature_idx in unique_nan_indices:
                nan_count = np.sum(np.isnan(features[:, feature_idx]))
                if verbose:
                    logging.info(f"Feature {feature_idx}: {nan_count} NaN values")
                # Get the row indices with NaN values for this feature
                row_indices_with_nan = nan_indices[0][nan_indices[1] == feature_idx]
                if verbose:
                    logging.info(
                        f"Row indices with NaN values for Feature {feature_idx}: {row_indices_with_nan}"
                    )

        # store the extracted features of this dataset in list
        feature_list.append(features)

        # get the feature names
        feature_names = list(radiomics_results[0].keys())

    mu_list = []
    sigma_list = []
    normalized_feature_list = []
    base_distribution = None
    if norm_sets_separately:
        # Concatenate all features to calculate the normalization statistics
        base_distribution = np.concatenate(feature_list, axis=0)
    for idx, features in enumerate(feature_list):
        if norm_type == "minmax":
            normalized_features = min_max_normalize(
                    features=features,
                    new_min=norm_range[0],
                    new_max=norm_range[1],
                    feature_names=feature_names,
                    base_distribution=base_distribution,
            )
        elif norm_type == "zscore":
            normalized_features = z_score_normalize(
                    features=features,
                    new_min=norm_range[0],
                    new_max=norm_range[1],
                    feature_names=feature_names,
                    base_distribution=base_distribution,
            )
        else:
            raise ValueError(
                f"Normalization type {norm_type} is not supported. "
                f"Please use 'minmax' or 'zscore'."
            )
        mu_list.append(np.mean(normalized_features, axis=0))
        sigma_list.append(np.cov(normalized_features, rowvar=False))
        normalized_feature_list.append(normalized_features)

    if save_features:
        for idx, normalized_features in enumerate(normalized_feature_list):
            # Extract the folder name from the first image file paths
            folder_name = Path(file_lists[idx][0]).parent.stem

            # Generate a unique identifier using the current timestamp
            unique_identifier = int(time.time())

            # Storage location
            storage_dir = str(Path(file_lists[idx][0]).parents[1])

            # Define the CSV file paths with a unique identifier and the folder name in the name.
            # Default: Store inside parent folder of image_paths
            csv_file_path = os.path.join(
                storage_dir,
                f"radiomics_set{idx}_results_{folder_name}_{unique_identifier}.csv",
            )
            norm_csv_file_path = os.path.join(
                storage_dir,
                f"radiomics_set{idx}_results_normalized_{folder_name}_{unique_identifier}.csv",
            )
            save_features_to_csv(csv_file_path, image_paths, mask_paths, radiomics_results)
            save_features_to_csv(norm_csv_file_path, image_paths, mask_paths, radiomics_results)

    return mu_list, sigma_list


def compute_statistics_of_paths(
    paths: list, # TODO
    norm_type: str,
    norm_range: list,
    feature_extractor,
    paths_mask: list=None,  # TODO
    resize_size=None,
    verbose=False,
    save_features=False,
    norm_sets_separately=True,
):
    """Calculates the statistics later used to compute the Frechet Distance for a given paths (i.e. one of the two distributions)."""

    if paths[0].endswith(".npz"):
        with np.load(paths[0]) as f:
            m, s = f["mu"][:], f["sigma"][:]
    else:
        file_lists = []
        mask_lists = []
        for path in paths:
            path = pathlib.Path(path)
            file_lists.append(sorted(
                [file for ext in IMAGE_EXTENSIONS for file in path.glob("*.{}".format(ext))])
            )
        if paths_mask is not None:
            for path_mask in paths_mask:
                if path_mask is None:
                    mask_lists.append(None)
                else:
                    path_mask = pathlib.Path(path_mask)
                    # Assumption: Each file in image dir has a corresponding file in mask dir with name
                    # similar enough to ensure correspondence via sorting
                    mask_lists.append(sorted(
                        [
                            mask
                            for ext in IMAGE_EXTENSIONS
                            for mask in path_mask.glob("*.{}".format(ext))
                        ]
                    ))
        else:
            mask_lists = [None, None]
            if verbose:
                logging.debug(f"file_lists in compute_statistics_of_path: {file_lists}")

        return calculate_feature_statistics(
            file_lists=file_lists,
            norm_type=norm_type,
            norm_range=norm_range,
            feature_extractor=feature_extractor,
            mask_lists=mask_lists,
            resize_size=resize_size,
            verbose=verbose,
            save_features=save_features,
            norm_sets_separately=norm_sets_separately,
        )


def get_feature_extractor(features, settings_dict: dict = None):
    """Returns a pyradiomics feature extractor allowing to customize the list of radiomics features to compute based on your dataset"""

    # Check if features is a string and a paths pointing to a csv file
    if isinstance(features, str) and features.endswith(".csv"):
        # raise a not implemented error to indicate that this feature is not yet implemented
        raise NotImplementedError(
            "Feature extraction based on a csv file is not yet implemented. "
            "Please open an issue on github if you would like us to add this feature:"
            "https://github.com/RichardObi/frd/issues/new/choose."
        )
        # with open(features, 'r') as f:
        #    features = [line.strip() for line in f]

    settings = {}
    if settings_dict is None:
        settings_dict = RADIOMICS_SETTINGS
    settings["setting"] = settings_dict
    # settings["setting"] = {"minimumROIDimensions": 1}

    # Set feature classes to compute
    settings["featureClass"] = {feature: [] for feature in features}
    return featureextractor.RadiomicsFeatureExtractor(settings)


def compute_frd(
    paths,
    features = ["firstorder", "glcm", "glrlm", "gldm", "glszm", "ngtdm", "shape", "shape2D"],
    norm_type = "minmax",
    norm_range = [0., 7.45670747756958],
    paths_masks=None,
    resize_size=None,
    verbose=False,
    save_features=True,
    norm_sets_separately=True,
):
    """Calculates the FRD based on the statistics from the two paths (i.e. the two distributions)
        Params:
    -- paths                : List of two paths to folders where images are stored representing the two distributions
    -- features             : The radiomics feature types to be extracted
    -- norm_type            : The method with which the extracted features should be normalized
    -- norm_range           : The range of normalization to scale the extracted features to after normalization
    -- paths_masks          : List of two paths to folders where segmentation masks are stored (name same as corresponding images).
    -- resize_size          : In case the images should be resized before the radiomics features are calculated
    -- verbose              : Indicates the verbosity level of the logging. If true, more info is logged to console.
    -- save_features        : Indicates whether the extracted features (original and normalized) should be saved to a csv file.
    -- norm_sets_separately : If true, indicates that the normalization should be done separately for the two sets of images.

    This function may be imported and called from other scripts to compute the FRD.
    """

    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError(f"Invalid paths: {p}")

    if not norm_sets_separately and '.npz' in paths[0]:
        raise ValueError(
            f"Normalization of datasets together is not supported when .npz file is provided. "
            f"In .npz file (normalized) statistics are already computed. "
            f"Please set norm_sets_separately to True or use image paths instead of .npz files."
        )

    feature_extractor = get_feature_extractor(features=features)

    mu_list, sigma_list = compute_statistics_of_paths(
        paths,
        norm_type,
        norm_range,
        feature_extractor,
        paths_mask=None if paths_masks is None else paths_masks,
        resize_size=resize_size,
        verbose=verbose,
        save_features=save_features,
        norm_sets_separately=norm_sets_separately,

    )

    if verbose: logging.debug(f"mu_list: {mu_list}, sigma_list: {sigma_list}")

    # Note: Assumption that len mu_list and len sigma_list is 2
    frd_value = calculate_frechet_distance(mu_list[0], sigma_list[0], mu_list[1], sigma_list[1])

    # Print this here instead of main() as compute_frd function may be used in other scripts as opposed to cmd line
    print(f"FRD: {frd_value}")

    return frd_value


def save_frd_stats(
    paths,
    features,
    norm_type: str,
    norm_range: list,
    paths_masks=None,
    resize_size=None,
    verbose=False,
    save_features=True,
):
    """Inits feature extractor creation and subsequent statistics computation and saving for the two distributions."""

    if not os.path.exists(paths[0]):
        raise RuntimeError(
            f"Please use a valid paths to imaging data. Currently got invalid paths: {paths[0]}"
        )

    if os.path.exists(paths[1]):
        raise RuntimeError(
            f"Please use an output file paths to an .npz file that does not yet exists. Currently got output file: {paths[1]}"
        )
    elif not paths[1].endswith(".npz"):
        logging.warning(
            f"Please revise as your provided stats output file paths '{paths[1]}' does not have an .npz extension. "
            f"Now continuing with the current paths."
        )

    feature_extractor = get_feature_extractor(features=features)

    if verbose:
        logging.info(f"Now computing and calculating statistics for {paths}")

    mu_list, sigma_list = compute_statistics_of_paths(
        [paths[0]],
        norm_type=norm_type,
        norm_range=norm_range,
        feature_extractor=feature_extractor,
        paths_mask=None if paths_masks is None else [paths_masks[0]],
        resize_size=resize_size,
        verbose=verbose,
        save_features=save_features,
    )

    np.savez_compressed(paths[1], mu=mu_list[0], sigma=sigma_list[0])


def main():
    args = parse_args()
    verbose = args.verbose
    if verbose:
        logging.info(args)

    if args.features is None:
        # we pass only one feature variable into the subsequent functions either containing a link (type str)
        # to a csv file or a list of feature names (type list)
        features = args.feature_groups
    else:
        features = args.features

    if args.save_stats:
        save_frd_stats(
            args.paths,
            features=features,
            norm_type=args.norm_type,
            norm_range=args.norm_range,
            paths_masks=args.paths_masks,
            resize_size=args.resize_size,
            verbose=args.verbose,
            save_features=args.save_features,
        )
        return

    frd_value = compute_frd(
        args.paths,
        features=features,
        norm_type=args.norm_type,
        norm_range=args.norm_range,
        paths_masks=args.paths_masks,
        resize_size=args.resize_size,
        verbose=args.verbose,
        save_features=args.save_features,
        norm_sets_separately=not args.norm_across,
    )
    # logging the result
    logging.info(
        f"Fréchet Radiomics Distance: {frd_value}. "
        f"Based on features: {features} with normalization type: {args.norm_type} and normalization range: {args.norm_range} (was normalization done separately for each dataset? -> {not args.norm_across}), with mask_lists: {args.paths_masks}, resized to f'{args.resize_size if args.resize_size is not None else ''}."
    )

if __name__ == "__main__":
    main()
