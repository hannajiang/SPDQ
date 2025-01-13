# SPDQ
The code of SPDQ: Synergetic Prompts as Disentanglement Queries for Compositional Zero-Shot Learning
## Setup
```
conda create --name xxx python=3.7
conda activate xxx
pip install torch torchvision torchaudio
```

## Download Dataset
Experiments are conducted on three datasets: MIT-States, UT-Zappos and C-GQA.
```
sh download_data.sh
```
## Training and Test
```
CUDA_VISIBLE_DEVICES=0 python -u train.py --yml.path xxxx.yml
```

If you want to test the trained model, please set --load_model is not None.

