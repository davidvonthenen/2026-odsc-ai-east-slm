# Building An SLM From Scratch

This project provides an end-to-end build of a Small Language Model (SLM) from scratch.

## Prerequisites

- Python 3.12+
- An 8 x H100 (or better single node system)

## Installation

```bash
# bring up your venv or (mini)conda
pip install -r requirements.txt
```

## Usage

### Step 1: Train Your Small Language Model (SLM) From Scratch 

By default this performs both the data ingest and training for our Small Laugnage Model (SLM).

```bash
python 1_train_8xH100.py
```

If you don't have access to that kind of hardware, download these prebuild models below.

PRIMARY DOWNLOAD:
https://drive.google.com/file/d/17thNDz5Y958JmG03utnwyUb8LX1LQbvh/view?usp=drive_link

BACKUP DOWNLOAD:
https://drive.google.com/file/d/1j0XsNlDdTHNy6c2YnVJd8eji5iOj58aB/view?usp=drive_link

### Step 2: Run Inference On Your Model

Download the model from the previous step (if needed), the run inference on your model.

```bash
python 2_inference.py
```
