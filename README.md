# Leaf Health Analyzer

Image classification pipeline for detecting plant leaf diseases using transfer learning (ResNet-50, EfficientNet-B0).

## Project Structure

```
├── data/
│   ├── plantvillage/       # PlantVillage lab images (place downloaded data here)
│   └── plantpathology/     # Plant Pathology 2021 field images (Kaggle)
├── models/                 # Saved checkpoints (.pt files)
├── notebooks/
│   ├── 00_eda.ipynb        # Exploratory data analysis
│   ├── 01_train_baseline.ipynb  # ResNet-50 baseline training
│   └── 02_ablation.ipynb   # Experiments: EfficientNet, focal loss, augmentation
├── src/
│   ├── data.py             # Dataset class and data loaders
│   ├── model.py            # Model wrappers (ResNet-50, EfficientNet-B0)
│   └── train.py            # Training utilities (loop, metrics, checkpointing)
├── requirements.txt
└── README.md
```

## Setup

The following setup is for Windows users using powershell. Please search up the equivalent way to set up a venv for Linux/Mac if you are not on Windows.

```bash
python -m venv venv
```

```bash
.\venv\Scripts\Activate.ps1
```

```bash
pip install -r requirements.txt
```

## Data

1. **PlantVillage**: Download from [Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) and extract to `data/plantvillage/`. Expected structure: `data/plantvillage/<class_name>/<image>.jpg`

```bash
kaggle datasets download -d abdallahalidev/plantvillage-dataset -p data/plantvillage --unzip
```

2. **Plant Pathology 2021**: Download from the [Kaggle competition](https://www.kaggle.com/competitions/plant-pathology-2021-fgvc8) and extract to `data/plantpathology/`. You must accept the competition rules on the Kaggle website before downloading.

```bash
kaggle competitions download -c plant-pathology-2021-fgvc8 -p data/plantpathology
```

Then unzip the downloaded file:

```powershell
Expand-Archive data/plantpathology/plant-pathology-2021-fgvc8.zip -DestinationPath data/plantpathology
```

Expected structure after extraction: `data/plantpathology/train_images/<image>.jpg` and `data/plantpathology/train.csv`

## Quickstart

Run `notebooks/00_eda.ipynb` first to verify data loading, then `notebooks/01_train_baseline.ipynb` to train and evaluate.

## References

- Hughes & Salathé (2015). PlantVillage dataset. arXiv:1511.08060
- Thapa et al. (2021). Plant Pathology 2021 – FGVC8. Kaggle.
- He et al. (2016). Deep Residual Learning for Image Recognition. CVPR.
- Tan & Le (2019). EfficientNet: Rethinking Model Scaling. ICML.
