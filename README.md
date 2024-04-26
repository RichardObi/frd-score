<!---[![PyPI](https://img.shields.io/pypi/v/frd.svg)](https://pypi.org/project/frd/)--->

# Fréchet Radiomics Distance (FRD)

This repository contains code implementing the FRD, proposed in [Towards Learning Contrast Kinetics with Multi-Condition Latent Diffusion Models](https://arxiv.org/abs/2403.13890).

FRD measures similarity of radiomics features between two datasets. 

<img src="docs/frd.png" alt="frd overview" width="400"/>

The lower the FRD, the more similar the datasets are in terms of radiomics features.

FRD is applicable to both _3D_ (nii.gz) and _2D_ (png, jpg, tiff) radiological images.

It is calculated by computing the [Fréchet distance](https://en.wikipedia.org/wiki/Fr%C3%A9chet_distance) between two Gaussians fitted to the extracted and normalized radiomics features.

In general, the variability (e.g. measured via FRD) of imaging biomarkers (e.g. radiomics features) between two datasets (e.g. a real and a synthetic dataset) can be interpreted as quality/utility metric (e.g. of a synthetic dataset).

## Installation

<!--- Install from [pip](https://pypi.org/project/frd/): --->
Install frd:

```
pip install git+https://github.com/RichardObi/frd
```

Requirements:
- python3
- pyradiomics
- SimpleITK
- pillow
- numpy
- opencv_contrib_python_headless
- scipy
- tqdm

## Usage

To compute the FID score between two datasets, where images of each dataset are contained in an individual folder:
```
python -m frd path/to/dataset_A path/to/dataset_B
```

If you would like to use masks to localize radiomics features, you can provide the path to the masks as follows:
```
python -m frd path/to/dataset_A path/to/dataset_B --paths_masks path/to/mask_A path/to/mask_B --is_mask_used
```

## Additional arguments

`--feature_groups`: You may define a subset of [radiomics features](https://pyradiomics.readthedocs.io/en/latest/customization.html#enabled-features) to calulate the FRD. Currently, a list of all features is used as default, i.e. `firstorder`, `glcm`, `glrlm`, `gldm`, `glszm`, `ngtdm`, `shape`, `shape2D`   

`--normalization_range`: The allowed value range of features in format `[min, max]`. Based on these values the frd features will be normalized. For comparability with FID, the default is `[0, 7.45670747756958]` which is an observed range for features of the Inception classifier in [FID](https://arxiv.org/abs/1706.08500). 

`--normalization_type`: The strategy with which the frd features will be normalized. Can be `minmax` or `zscore`.

`--normalize_across_datasets`: If set, indicates that normalization will be computed on all features from both datasets (e.g. synthetic, real) instead of on the features of each dataset separately.

`--resize_size`: You may indicate an integer here to resize the x and y pixel/voxel dimensions of the input images (and masks) using `cv2.INTER_LINEAR` interpolation. For example `resize_size=512` will resize an image of dims of e.g. `(224, 244, 120)` to `(512, 512, 120)`.

`--save_features`: Indicates whether radiomics feature values (normalized and non-normalized) should be stored in a csv file in the parent dir of `path/to/dataset_A`. This can be useful for reproducibility and interpretability.

`--is_mask_used`: If set, indicates that radiomics features are generated based on the available mask instead of the whole input image.

`--verbose`: You may enable more detailed logging.info and logging.debug console logs by providing the `verbose` argument.

`--save-stats`:
As in [pytorch-fid](https://github.com/mseitzer/pytorch-fid), you can generate a compatible `.npz` archive of a dataset using the `--save-stats` flag. 
You may use the `.npz` archive as dataset path, which can be useful to compare multiple models against an original dataset without recalculating the statistics multiple times.
```
python -m pytorch_fid --save-stats path/to/dataset path/to/outputfile
```


## Citing

If you use this repository in your research, consider citing it using the following Bibtex entry:
```
@article{osuala2024towards,
  title={{Towards Learning Contrast Kinetics with Multi-Condition Latent Diffusion Models}},
  author={Osuala, Richard and Lang, Daniel and Verma, Preeti and Joshi, Smriti and Tsirikoglou, Apostolia and Skorupko, Grzegorz and Kushibar, Kaisar and Garrucho, Lidia and Pinaya, Walter HL and Diaz, Oliver and others},
  journal={arXiv preprint arXiv:2403.13890},
  year={2024}
```

## Acknowledgements

An initial implementation was provided by [Preeti Verma](https://github.com/preeti-verma8600).

This repository borrows code from the [pytorch-fid](https://github.com/mseitzer/pytorch-fid) repository, the official pytorch implementation of the [Fréchet Inception Distance](https://arxiv.org/abs/1706.08500).
