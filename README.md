# Exploring Cross-Modal Flows for Few-Shot Learning

Official implementation of the paper [Exploring Cross-Modal Flows for Few-Shot Learning](https://arxiv.org/abs/2510.14543).


## TODO 
- FMA framework. ✅
- Support CLIP extractor. ✅
- Other extractor and checkpoints: coop, cocoop, lora, adapter.


## Table of Contents

- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Citation](#citation)

## ⚙️ Installation

### Prerequisites

- Python 3.8.20
- Pytorch 2.3.0
- CUDA 12.1 

### Environment Setup




```bash
git clone https://github.com/HKUST-LongGroup/FMA.git
cd FMA
```


```bash
conda create -n fma python=3.8.20
conda activate fma
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install git+https://github.com/openai/CLIP.git
pip install scipy==1.10.1
```

## 📁 Dataset Preparation


### Setup Instructions

1. Create a data directory:
```bash
mkdir -p data
```

2. Follow the detailed instructions in [DATASETS.md](DATASETS.md) to download and organize each dataset.

3. The expected directory structure:
```
data/
├── oxford_pets/
├── eurosat/
├── ucf101/
├── sun397/
├── caltech-101/
├── dtd/
├── fgvc_aircraft/
├── food-101/
├── oxford_flowers/
├── stanford_cars/
└── imagenet/
```

## 🚀 Training

###  Basic Training

Train a model with default configuration (EuroSAT, 16-shot, CLIP ViT-B/16):

```bash
python train.py
```

### Custom Configuration

You can customize the training by specifying command-line arguments:

```bash
# Train on a specific dataset
python train.py --dataset OxfordPets

# Specify number of shots
python train.py --dataset EuroSAT --num_shots 8

# Choose feature extractor
python train.py --feature_extractor clip

# Combine multiple options
python train.py --dataset UCF101 --num_shots 4 --feature_extractor clip
```

### Available Arguments

- `--dataset`: Dataset name (default: `EuroSAT`)
  - Options: `OxfordPets`, `EuroSAT`, `UCF101`, `SUN397`, `Caltech101`, `DescribableTextures`, `FGVCAircraft`, `Food101`, `OxfordFlowers`, `StanfordCars`, `ImageNet`
  
- `--num_shots`: Number of shots for few-shot learning (default: `16`)
  - Options: `1`, `2`, `4`, `8`, `16`
  
- `--feature_extractor`: Feature extractor type (default: `clip`)
  - Options: `clip`, `coop`, `cocoop`

### Configuration File

You can also modify the default configuration in `config.py`:

```python
class DefaultConfig:
    def __init__(self):
        self.epochs = 600
        self.batch_size = 32
        self.lr = 2e-4
        self.clip_type = 'ViT-B/16'  # CLIP backbone
        self.dataset = 'EuroSAT'
        self.num_shots = 16
        self.blocks = 12  # Number of residual blocks
        # ... other parameters
```

### Training Output

Training creates a timestamped checkpoint directory:
```
checkpoints/
└── 20251021_010857/
    ├── config.json      # Training configuration
    ├── model.pth        # Trained model weights
    └── log.txt          # Training logs
```

## 🧪 Evaluation

### Test a Trained Model

To evaluate a trained model, use the timestamp of the checkpoint:

```bash
python test.py <timestamp>
```

Example:
```bash
python test.py 20251021_010857
```

This will:
- Load the saved configuration and model weights
- Evaluate the model on the test set with different inference steps (0-10)
- Report accuracy for each number of steps


## 🤖 Project Structure

```
.
├── config.py                  # Configuration and hyperparameters
├── train.py                   # Training script
├── test.py                    # Evaluation script
├── models/
│   ├── fm.py                  # Flow matching network
│   ├── feature_extractor.py  # Feature extraction interface
│   └── clip_extractor.py     # CLIP feature extractor
├── datasets/
│   ├── __init__.py           # Dataset registry
│   ├── eurosat.py            # EuroSAT dataset
│   ├── oxford_pets.py        # Oxford Pets dataset
│   └── ...                   # Other datasets
├── checkpoints/              # Saved models 
├── data/                     # Datasets 
├── DATASETS.md              # Dataset preparation guide
└── README.md                # This file
```


## Acknowledgments

- [CLIP](https://github.com/openai/CLIP) for pre-trained vision-language models
- [CoOp](https://github.com/KaiyangZhou/CoOp) for dataset preparation scripts


## Citation
```
@misc{jiang2025exploringcrossmodalflowsfewshot,
      title={Exploring Cross-Modal Flows for Few-Shot Learning}, 
      author={Ziqi Jiang and Yanghao Wang and Long Chen},
      year={2025},
      eprint={2510.14543},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.14543}, 
}
```