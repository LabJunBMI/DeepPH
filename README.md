# Code for [DeepPH]

## File Descriptions

- `data_split.py`  
  Script for preprocess ing enzyme data into graph format with multimodal features for pH range prediction.

- `egnn_clean.py`  
  Defines the core EGNN architecture used in our method.

- `egnn_model_split_range.py`  
  The full model pipeline for training on range prediction tasks.

- `train_model_value_split_range.py`  
  Main training script for learning both the pH value and its interval (range).  
  Supports GPU training and logs intermediate metrics.

- `test_model_r2_split_aa_range3.pt`  
  The trained model containing learned weights.

- ‘new_train_value.pkl, new_test_value.pkl and new_test_value_remove_phenv.pkl'
  They are our training dataset and two test sets.

## How to Run

### 1. Environment Setup

```bash
conda create -n deepph python=3.8
conda activate deepph
pip install -r requirements.txt
```

### 2. Train the Model

To train DeepPH from scratch:

```bash
python train_model_value_split_range.py
```


### 3. Evaluate a Trained Model (No Retraining Required)

To evaluate a trained model checkpoint directly, without retraining, use `test.py`:

```bash
python test.py \
    --model_path test_model_r2_split_aa_range3.pt \
    --test_data new_test_value.pkl
```


## Analysis

This repository also provides a Jupyter notebook, `analysis.ipynb`, which 
reproduces the residue-level attention analyses presented in the manuscript, 
including sequence-level attention patterns across amino acid types, 
structure-level analyses linking attention to geometric and conformational 
features, and functional validation of attention against known catalytic 
active sites.

To run the notebook:

```bash
jupyter notebook analysis.ipynb
```

The notebook assumes a trained model checkpoint and a test set are 
available in the working directory, and extracts per-residue attention 
weights by running a forward pass of the trained model over the test set 
before generating each analysis.

Three-dimensional visualizations mapping attention onto protein structures 
are provided separately using PyMOL.
