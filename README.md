# Skin Tone Prediction from Dermatoscopic Images

Code and data for the paper TODO. TODO: Add link to arXiv

Colorimeter-Supervised Skin Tone Estimation from Dermatoscopic Images for Fairness Auditing, Marin Benčević, Krešimir Romić, Ivana Hartmann Tolić, Irena Galić

## Dependencies

This project uses the `uv` runner to execute scripts and manage common tasks. All Python package requirements are listed in `pyproject.toml` so you can inspect the exact dependencies there.

Follow the `uv` documentation for installation and usage: https://github.com/astral-sh/uv

## Training data

Training data is organized in `data/<dataset_name>`, the same dataset name is used in the configs and test/inference script options.

Available dataset names for the configs and script options: `diverse`, `fp17k`, `scin`, `pad_ufes_20`, `mra-midas`, `ssynth` (note: stored in `data/output10k`), `mskcc`.

Data should be stored as follows:

```
data/
  diverse/
    000001.png
    ...
    ddi_metadata.csv
  fp17k/
    images/
      0a0e21f413499ad85018f7fa0df3efe2
      ...
    fitzpatrick17k.csv
  mra-midas/
    release_midas.xlsx
    s-prd-396524710.jpg
    ...
  mskcc/
    images/
      ISIC_0077599.jpg
      ...
    s1.csv
    ...
    s8.csv
    metadata.csv
  output_10k/output/
    skin000/
    ...
  pad_ufes_20/zr7vgbcyr2-1/
    images/
      imgs_part_1/
      ...
    metadata.csv
  scin/dataset/
    images/
      -72454742289752.png
      ...
    scin_cases.csv
    scin_labels.csv
```

### Dataset links

| dataset_name | Dataset | Link |
|---|---|---|
| diverse | Diverse | https://ddi-dataset.github.io/ |
| fp17k | Fitzpatrick 17k | https://github.com/mattgroh/fitzpatrick17k |
| pad_ufes_20 | PAD UFES 20 | https://www.sciencedirect.com/science/article/pii/S235234092031115X |
| scin | SCIN | https://github.com/google-research-datasets/scin |
| ssynth | SSYNTH (output10k.zip) | https://huggingface.co/datasets/didsr/ssynth_data/tree/main/data/synthetic_dataset |
| mra-midas | MRA-MIDAS | https://doi.org/10.71718/15nz-jv40 |
| mskcc | MSKCC | https://doi.org/10.34970/962049 |

## Training

Training is based on `config/<experiment_name>.json` which results in `experiments/<experiment_name>` folder where logs and checkpoints are stored. Uses 5-fold CV by default; check the configs for more info.

Main configs used in the paper:

- `pretrain_blur0.json` — pretraining efficientnet_b0 for all downstream models

Finetuning configs rely on the above being trained.

Fine-tuning configs:

- `finetune_mskcc_blur0.json` — fine-tuning FP ordinal regression (main FP prediction network)
- `finetune_mskcc_blur0_lab.json` — fine-tuning the CIELab regression (i.e. ITA prediction) network

Additional experiments for the paper (worse performance):

- `finetune_mskcc_blur0_wb.json` — with shades of gray white balancing
- `finetune_mskcc_blur0_classification.json` — classification instead of ordinal regression for FP prediction (uses `train_fp_classification.py` and `test.py --mode classification`)

### How to run training

Pretrain backbone for CORAL and LAB regression:

```
uv run train_fp_regression.py configs/pretrain_blur0.py
```

Pretrain backbone for regular FP classification head (worse performance):

```
uv run train_fp_classification.py configs/pretrain_classification.json
```

**Fine tuning:**

(Note: in the config you can specify which experiment name to use as the checkpoint to initialize the network)

Fitzpatrick CORAL ordinal regression head:

```
uv run train_fp_regression.py configs/finetune_mskcc_blur0.json
```

Testing:

```
uv run test.py configs/finetune_mskcc_blur0.json
```

**Fitzpatrick classification head (worse performance):**

```
uv run train_fp_classification.py configs/finetune_mskcc_blur0_classification.json
```

Testing:

```uv run test.py configs/finetune_mskcc_blur0_classification.json --mode classification```

## Config explanation

- `experiment_name`: name to store in `experiments/<experiment_name>`
- `model`: supported values are `efficientnet_b4_coral` (CORAL head FP prediction), `efficientnet_b4_classification` (regular FP classification head), `efficientnet_b4_lab` (CIELab regression with delta E loss)
- `checkpoint`: null or `experiment_name` to use as initialization
- `dataset_name`: which datasets to use for training, can add multiple datasets with `+` as separator, e.g. `diverse+fp17k+scin`
- `blur_amount`: sigma how much to blur the image, recommended 0 to disable blur
- `learning_rate`
- `num_epochs`
- `batch_size`
- optional: `k_folds`: number of folds to use for k-fold, recommended 5 for fine-tuning; if omitted a regular 80/10/10 split will be used

## How to run inference scripts

### Download links and citations

ISIC 2020 - International Skin Imaging Collaboration. SIIM-ISIC 2020 Challenge Dataset. International Skin Imaging Collaboration https://doi.org/10.34970/2020-ds01 (2020).

ISIC 2020 test: https://api.isic-archive.com/collections/68/

MILK 10k MILK study team. MILK10k. ISIC Archive, 2025, doi:10.34970/648456. https://challenge.isic-archive.com/data/#milk10k

Download the datasets with the following file structure:

```
inference_data/
  isic2020/
    train-image/image/
      ISIC_0015719.jpg
      ...
    train-metadata.csv
    ISIC_2020_Training_GroundTruth_v2.csv
  isic2020_test/
    test-resized/
      ISIC_0052060.jpg
      ...
    challenge-2020-test_metadata_2026-01-14.csv
  milk10k/
    MILK10k_Training_Input/
      IL_0000652/
      ...
    MILK10k_Training_GroundTruth.csv
    MILK10k_Training_Metadata.csv
```

Run lab inference:

```
uv run lab_inference.py <dataset_name> <config>
```

Supported dataset names: `isic2020`, `isic2020_test`, `milk10k`

Example:

```
uv run lab_inference.py isic2020 configs/finetune_mskcc_blur0_lab.json
```

## How to get the image-based ITA results

See `get_ita_for_data.py` for image-based ITA computations (k-Means / path-based methods) on processed images.

## Adding a new inference dataset

Requires a dataset to be defined in `inference_dataset.py`.

How to add a new one:

In the `InferenceDatasetMixin` add a new function called `<dataset_name>_init` and set `self.orig_files` to be a list of file paths, `self.patient_ids` to be the patient ID for each file, and call `self.ensure_processed_images(self.orig_files, self.processed_dir)` to resize and store the resized images before training.

```python
    def isic2020_init(self, files):
        metadata_path = 'inference_data/isic2020/train-metadata.csv'
        images_dir = 'inference_data/isic2020/train-image/image'
        df = pd.read_csv(metadata_path)
        df['image_path'] = df['isic_id'].apply(lambda x: os.path.join(images_dir, f"{x}.jpg"))
        df = df[df['image_path'].apply(os.path.exists)]
        if files is not None:
            df = df[df['image_path'].isin(files)]
        self.orig_files = df['image_path'].tolist()
        self.patient_ids = df['patient_id'].tolist()
        self.ensure_processed_images(self.orig_files, self.processed_dir)
```

## Pretrained models

The pretrained models are available in the Releases section on GitHub. Place the folders in `experiments/`.

For example:

```
experiments/finetune_mskcc_blur0/
  fold_0/
  ...
  tensorboard/
  config.json
```

## Reproducing analysis

All plots, figures and tables in the paper are produced by notebooks in this repository.

- `analyze_inference_results.ipynb` - ISIC 2020 and MILK 10k distribution
- `analyze_test_results.ipynb` - Metrics and analysis of the model performance
