# Out-of-Domain (OOD) Detection

FRD can detect whether newly acquired medical images come from the same domain as a reference set. This is useful for flagging potential distribution shifts — for example, images acquired with a different scanner or protocol.

## Image-level OOD detection

Per-image scoring computes the L2 distance of each test image's radiomic feature vector from the reference set mean, then uses a statistical threshold to classify each image as in-domain or out-of-domain.

### CLI

```bash
python -m frd_score ood path/to/reference path/to/test_images
```

### Python API

```python
from frd_score import compute_frd, detect_ood

# First, extract normalised features
# (detect_ood expects pre-normalised feature arrays)
# Typically you'd get these from compute_statistics_of_paths or compute_features

results = detect_ood(
    feature_list=[ref_features, test_features],
    detection_type="image",
    seed=42,
    filenames=test_filenames,
)

print(f"Threshold: {results['threshold']:.4f}")
print(f"OOD detected: {results['predictions'].sum()}/{len(results['predictions'])}")
```

### Output

Results are saved to `outputs/ood_predictions/ood_predictions.csv`:

| Column | Description |
|---|---|
| `filename` | Image filename (or index) |
| `ood_score` | L2 distance from reference mean |
| `ood_prediction` | `True` = OOD, `False` = in-domain |
| `p_value` | Probability of being in-domain |

## Dataset-level OOD detection (nFRD)

The dataset-level score estimates the fraction of an entire dataset that is out-of-domain, using the area under the ROC curve.

### CLI

```bash
python -m frd_score ood path/to/reference path/to/test_images --detection_type dataset
```

### Python API

```python
results = detect_ood(
    feature_list=[ref_features, test_features],
    detection_type="dataset",
)

print(f"nFRD: {results['nfrd']:.4f}")
```

The `nfrd` score ranges from 0 (completely in-domain) to 1 (completely out-of-domain).

## Configuration

### Statistical assumptions

The `--id_dist_assumption` flag controls how the threshold is estimated from the reference scores:

| Value | Description |
|---|---|
| `gaussian` (default) | Assume in-distribution scores follow a Gaussian; threshold at 95th percentile |
| `t` | Use Student's t-distribution with DoF = n−1 |
| `counting` | Non-parametric; threshold at empirical 95th percentile |

### Validation split

By default, the full reference set is used for both the mean embedding and threshold estimation. To hold out a validation subset:

```bash
python -m frd_score ood path/to/ref path/to/test \
    --use_val_set \
    --val_frac 0.2
```

This uses 20% of the reference images for threshold estimation and the remaining 80% for computing the reference mean.

### Reproducibility

Use `--seed` for deterministic results:

```bash
python -m frd_score ood path/to/ref path/to/test --seed 42
```
