# TRICARE

### Deep-learning triage of 3D pathology datasets for comprehensive and efficient pathologist assessments

Gan Gao, Renao Yan, Andrew H. Song, Huai-Ching Hsieh, Lindsey A. Erion Barner, ..., Faisal Mahmood, Jonathan T.C. Liu  
*Nature BME*

[[Paper]](https://www.nature.com/articles/s41551-026-01760-1) | [[Preprint]](https://www.biorxiv.org/content/10.1101/2025.07.20.665804v1) | [[Video]](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41551-026-01760-1/MediaObjects/41551_2026_1760_MOESM4_ESM.mp4) | [[Dataset]](https://zenodo.org/records/20052262)

<img src=images/Overview.png>

## Overview

This repository contains code for training **TRICARE**, a deep learning triage framework that identifies high-risk 2D cross sections within large 3D pathology datasets to enable time-efficient pathologist evaluation. **TRICARE** leverages context from a subset of neighboring depth levels, achieving better performance than models that learn solely from isolated 2D levels. Please refer to our [paper](https://www.nature.com/articles/s41551-026-01760-1) for more details.

## Updates
**(08/12/26)** The TRICARE article is now published in _**Nature BME**_! The codebase will be continually updated.

## Table of Contents
- [Install](#Install)
- [Usage](#usage)
  - [Step 1: Create Data Splits](#step-1-create-data-splits)
  - [Step 2: Train the Model](#step-2-train-the-model)
- [Full 3D data](#full-3d-data)


---

## Install

Install the required packages using the following commands. This process may take a few minutes to complete.

```bash
conda env create -f environment.yml
```

Activate virtual environment

```bash
conda activate tricare_codes
```

---

## Usage

### Step 1: Create Data Splits

First, place your CSV file under the `dataset_csv` directory. We've provided an example spreadsheet. 

Next, update the `csv_path` variable within the `Generic_WSI_Classification_Dataset` class in the `create_splits_seq.py` file.

Run

```bash
python create_splits_seq.py \
    --seed 3 \
    --k 8 \
    --leave_one_out
```

--seed: Random seed for reproducibility.

--k: Number of folds.

--leave_one_out: If set, performs leave-one-out validation. `--k` should then be set to the number of patients (one fold per held-out patient).

### Step 2: Train the Model

First, generate patch-level features and save them using the naming convention `sample_depth.pt` (e.g., `BiopsyA-a_001.pt`). We provide example feature files in the [TRICARE test_data shared drive](https://drive.google.com/drive/folders/1KRFZ9tURuyMOjGMvZ7XJy0jg2Gzj54La?usp=sharing). Please download and place them under 'test_data'.

In this work, we found that using the **CONCH** model ([Nature Medicine, 2024](https://www.nature.com/articles/s41591-024-02856-4)) yields better performance. Users can follow the instructions in the official [CONCH GitHub repository](https://github.com/mahmoodlab/CONCH) to generate features from histology patches.

Next, update the `csv_path` variable within the `Generic_WSI_Classification_Dataset` class in the `main.py` file.

Run

```bash
CUDA_VISIBLE_DEVICES=1 python main.py \
    --drop_out \
    --lr 2e-4 \
    --k 8 \
    --leave_one_out \
    --agg_range 3 \
    --agg_gap 3 \
    --adj_gap 5 \
    --exp_code exp_prostate_range60gap60 \
    --weighted_sample \
    --max_epochs 50 \
    --bag_loss ce \
    --model_type carp3d_ld \
    --log_data \
    --data_root_dir test_data/

```

--drop_out: Enable dropout in model.

--lr: Learning rate.

--k: Number of folds.

--leave_one_out: Use leave-one-out split. `--k` should equal the number of patients when this is set (see Step 1).

--agg_range: Maximum range of levels above and below the target depth for 2.5D aggregation. For example, with --agg_range 3, the model aggregates features from up to 3 levels above and 3 below the target level.

--agg_gap: Step size (in depth levels) between levels for 2.5D aggregation. For example, with --agg_gap 3, the model uses every 3 levels within agg_range.

--adj_gap: The depth between adjacent levels as in the file names.

--exp_code: Directory name for saving results.

--weighted_sample: Use class-balanced sampling.

--max_epochs: Max training epochs.

--bag_loss: Loss function.

--model_type: Model architecture to use.
* `abmil`: 2D ABMIL — baseline attention-based MIL over a single depth level, with no cross-depth context.
* `carp3d_naive`: TRICARE Naive — concatenates patches from every level in the aggregation window and attends over them jointly, without a separate depth-level structure.
* `carp3d_dl`: TRICARE D->L (Depth→Lateral) — attends across depth first at each patch position, then laterally across patches. Requires every depth level in the aggregation window to have the same number of patches, at matching spatial coordinates, since patch *i* at one level is assumed to correspond to patch *i* at every other level.
* `carp3d_ld` (default): TRICARE L->D (Lateral→Depth) — attends laterally across patches within each depth level first, then across depth levels via a learned attention.
* `carp3d_ld_ave`, `carp3d_ld_linear_attn`, `carp3d_ld_rnn`: Other L->D variants that aggregate across depth via simple averaging, linear attention, or a bidirectional RNN, respectively, instead of gated attention.

--log_data: Record log data with tensorboard.

--data_root_dir: Path to patch features.
 
We thank [CLAM GitHub repository](https://github.com/mahmoodlab/CLAM) for the computation framework.

---

## Full 3D data

We provide the 3D pathology images used in the prostate and esophagus computational experiments, including both development and independent test cohorts for each use case, via [Zenodo](https://doi.org/10.5281/zenodo.20052262). Binary annotations are provided in the accompanying spreadsheets. Users can follow the setup instructions above to train on this data. Training on the entire prostate and esophagus cohorts, using the hardware and hyperparameters specified in our paper, takes a few hours.

The full 3D pathology datasets — beyond the images used for the computational experiments — for the prostate development cohort are additionally available from the TCIA Prostate 3D Pathology Collection at https://www.cancerimagingarchive.net/collection/pca_bx_3dpathology/.

To extract the levels from these full 3D pathology datasets, run the `extract_levels.py` script. Then follow the [falsecolor-python algorithm](https://github.com/serrob23/falsecolor) to render H\&E-like images from fluorescence data.

Additional 3D pathology datasets from this research program are available for research and educational use. We will update the Dataset link above once these are released on TCIA.

---

## Contact

For any suggestions or issues, please contact Gan Gao (gangao@uw.edu).

## Cite

If you find our work useful in your research or if you use parts of this code, please cite our paper:

> Gao, G., Yan, R., Song, A.H. et al. Deep-learning triage of three-dimensional pathology datasets for comprehensive and efficient pathologist assessments. *Nat. Biomed. Eng* (2026). https://doi.org/10.1038/s41551-026-01760-1
