# Deep-learning triage of 3D pathology datasets for comprehensive and efficient pathologist assessments

This repository contains code for training **TRICARE**, a deep learning triage framework that identifies high-risk 2D cross sections within large 3D pathology datasets to enable time-efficient pathologist evaluation. **TRICARE** leverages context from a subset of neighboring depth levels, achieving better performance than models that learn solely from isolated 2D levels.

## Table of Contents
- [Install](#Install)
- [Usage](#usage)
  - [Step 1: Create Data Splits](#step-1-create-data-splits)
  - [Step 2: Train the Model](#step-2-train-the-model)
- [Full 3D data](#full-3d-data)


---

## Install

Install the required packages using:

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

--leave_one_out: If set, performs leave-one-out validation.

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

--leave_one_out: Use leave-one-out split.

--agg_range: Maximum range of levels above and below the target depth for 2.5D aggregation. For example, with --agg_range 3, the model aggregates features from up to 3 levels above and 3 below the target level.

--agg_gap: Step size (in depth levels) between levels for 2.5D aggregation. For example, with --agg_gap 3, the model uses every 3 levels within agg_range.

--adj_gap: The depth between adjacent levels as in the file names.

--exp_code: Directory name for saving results.

--weighted_sample: Use class-balanced sampling.

--max_epochs: Max training epochs.

--bag_loss: Loss function.

--model_type: Model architecture to use .

--log_data: Record log data with tensorboard.

--data_root_dir: Path to patch features.
 
We thank [CLAM GitHub repository](https://github.com/mahmoodlab/CLAM) for the computation framework.

---

## Full 3D data

The prostate model used in this work was developed using our publicly available data from TCIA Prostate 3D Pathology Collection at https://www.cancerimagingarchive.net/collection/pca_bx_3dpathology/.

To extract the levels annotated by pathologists, run the `extract_levels.py` script using the data provided in `dataset_csv/full_prostate_labels.csv`. Then follow the [falsecolor-python algorithm](https://github.com/serrob23/falsecolor) to render H\&E-like images from fluorescence data.

Once preprocessing is complete, you can proceed to train the model using all images from the prostate development cohort.

Due to the large size of the datasets (tens of terabytes), we are working with platforms such as the NIH TCIA to host and publicly release the remaining datasets and associated clinical data following the publication of the manuscript.
