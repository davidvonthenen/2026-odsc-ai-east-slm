# Building An Masked Language Model (MLM) Using Distillation Training

This project provides an end-to-end build of a Masked Language Model (MLM) using Distillation Training.

## Prerequisites

- Python 3.12+
- An H100 (or better)

## Installation

```bash
# bring up your venv or (mini)conda
pip install -r requirements.txt
```

## Usage

### Step 1: Data Prep

This step will perform data prep for training.

```bash
python 1_prepare_data.py
```

### Step 2: Train Your MLM

This will take the data and train your MLM.

```bash
python 2_train_distill_loop.py
```

If you don't have access to that kind of hardware, download these prebuild models below.

PRIMARY DOWNLOAD:
https://drive.google.com/file/d/1SYCi8zk8oe3TCLUg9rRuFpXrUQblV9xE/view?usp=drive_link

BACKUP DOWNLOAD:
https://drive.google.com/file/d/1vQUdwhs-bWvsJtdhp1DKjhF-30Allpcb/view?usp=drive_link

### Step 3: Run Inference On Your Model

Download the model from the previous step (if needed), the run inference on your model.

```bash
python 3_inference.py
```
