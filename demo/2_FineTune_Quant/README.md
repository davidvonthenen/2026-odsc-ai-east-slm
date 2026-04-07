# Building An SQL-based SLM Using Fine-Tuning (and Quantization)

This project provides an end-to-end build of a SQL-based Small Language Model (SLM) via fine-tuning and quantization.

## Prerequisites

- Python 3.12+
- An H100 (or better)

## Installation

```bash
# bring up your venv or (mini)conda
pip install -r requirements.txt
```

## Usage

### Step 1: Data Prep + Fine-Tune A Model (Qwen 2.5)

This step will perform data prep and fine-tune your model.

```bash
python 1_finetune.py
```

If you don't have access to that kind of hardware, download these prebuild models below and skip to `Step 3`.

PRIMARY DOWNLOAD:
[https://drive.google.com/file/d/1rLUd1N6mUkaTGtt6Pv5JB4MtgEkufPoA/view?usp=drive_link](https://drive.google.com/file/d/1rLUd1N6mUkaTGtt6Pv5JB4MtgEkufPoA/view?usp=drive_link)

BACKUP DOWNLOAD:
[https://drive.google.com/file/d/1n1jsVjuYYIEpWvb4KPnlFoSpuY1T9byJ/view?usp=drive_link](https://drive.google.com/file/d/1n1jsVjuYYIEpWvb4KPnlFoSpuY1T9byJ/view?usp=drive_link)

### Step 2: Quantize Your Fine-Tuned Model

Take the model you trained in step 1 above and quantize it.

#### Apple Silicon Only Instructions

If and only if you have MacOS with Apple Silicon, use the version with `-MLX` in the filename.

```bash
# Apple only!
python 2_quantize-MLX.py
```

#### Instructions for Everyone Else (aka All non-Apple Silicon)

You will need a copy of `llama.cpp` in this folder and it's tools to perform a GGUF quantization.

```bash
# get local llama.cpp copy
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
# checkout latest release as of writing this
git checkout b8762
```

Donwload the prebuilt `llama.cpp` binaries for this release and drop everything from the extracted archive into the `llama.cpp` folder.

[https://github.com/ggml-org/llama.cpp/releases/tag/b8762](https://github.com/ggml-org/llama.cpp/releases/tag/b8762)

To quantize using GGUF, use the version with `-CPU` in the filename.

```bash
# Everyone else
python 2_quantize-CPU.py
```

### Step 3: Run Inference On Your Model

Run inference on your model.

```bash
# Apple only!
python 3_inference-MLX.py
```

For all other types of computers, use  the version with `-CPU` in the filename.

```bash
# Everyone else
python 3_inference-CPU.py
```
