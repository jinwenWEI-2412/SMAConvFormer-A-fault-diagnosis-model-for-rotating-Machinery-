# SMAConvFormer: A fault diagnosis model for rotating machinery with high noise and variable operating conditions

SMAConvFormer — a lightweight PyTorch framework for fault diagnosis of rotating machinery. It provides data preprocessing, multiple model implementations (including the proposed SMAConvformer), and training/evaluation pipelines so researchers can reproduce experiments or run comparisons on similar datasets.

## Quick summary
- Paper: "SMAConvFormer: a fault diagnosis model for rotating machinery with high noise and variable operating conditions" (Measurement Science and Technology)
- Python: 3.8 recommended
- Framework: PyTorch (see requirements)

## Quick start (minimum steps)
1. Create virtual environment and install dependencies
```bash
python3.8 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Prepare dataset
- Download one of the datasets listed below and place the extracted files under `./data/<dataset_name>/` as described in each dataset script, or download the provided "save dataset" (a preprocessed .pkl) and put it in `./data/save_dataset/`.

3. Configure run
- By default the project reads CLI args via `config/args_config.py` and `args_diagnosis.py`.
- Example: to train SMAConvformer on BJTU_rao with defaults, edit `config/args_config.py` or run from command line (see examples below).

4. Run training
```bash
# default entrypoint
python train.py

# example with explicit args (override defaults)
python -m train --model_name SMAConvformer --dataset_name BJTU_rao --batch_size 32 --lr 0.01 --epoch 100
```

Notes:
- If you want to first create the preprocessed dataset using the raw data, set `--save_dataset True` and ensure dataset scripts can find the raw files. Example:
```bash
python -m train --save_dataset True --dataset_name BJTU_rao
```
This will create `./data/save_dataset/<dataset_name>.pkl` used by the data loaders.

## Datasets (links in original README)
- Case1: XJTU gearbox
- Case2: XJTU spurgear
- Case3: OU bearing
- Case4: BJTU-Rao
- Save dataset (preprocessed .pkl)

See `datasets/` for per-dataset loader and preprocessing details.

## How to run common scenarios
- Train (default args in config/args_config.py)
  ```bash
  python train.py
  ```
- Train with specific model and lr
  ```bash
  python -m train --model_name SMAConvformer --dataset_name BJTU_rao --lr 0.01 --epoch 120
  ```
- Create preprocessed dataset from raw files
  ```bash
  python -m train --save_dataset True --dataset_name XJTU_gearbox
  ```
- Test only (load checkpoint saved under `results/<dataset_name>/`)
  ```bash
  python -m train --only_test True --dataset_name BJTU_rao
  ```

## Repo layout (high level)
- models/      model implementations (SMAConvformer, LiConvFormer, EWSNet, baselines)
- datasets/    dataset loaders & preprocessing for XJTU, OU, BJTU
- config/      CLI arg definitions and helpers
- utils/       training/validation/test loop, logging, and helpers
- train.py     training entrypoint

## Tips
- Python 3.8 and PyTorch >= 1.9.0 are recommended. README originally used torch 1.10.1.
- If you encounter errors when saving data, try lowering numpy version (README suggests using numpy 1.22.0 may cause issues; try 1.21.x or 1.20.x).
- Logs, checkpoints and predictions are saved under `./results/<dataset_name>/`.

## Citation
If this repository helps your research, please cite:
```
@paper{
  title = {SMAConvFormer: a fault diagnosis model for rotating machinery with high noise and variable operating conditions},
  author = {Jingwen Wei, Jianhai Yue},
  journal = {Measurement Science and Technology},
  year = {2026},
}
```
