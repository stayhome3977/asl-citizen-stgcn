# Isolated ASL Sign Recognition from MediaPipe Keypoints (ST-GCN)

Recognition of 20 isolated American Sign Language signs from the [ASL Citizen](https://www.microsoft.com/en-us/research/project/asl-citizen/) dataset, using MediaPipe Holistic keypoint sequences and a skeleton-based ST-GCN trained from scratch. Course project (Deep Learning), team of 5. This repository contains the final ST-GCN training and evaluation notebook.

## Task and data

- 20 ASL Citizen signs, one clip per sample, 64 frames per clip.
- 102 features per frame: x,y of both hands (21 landmarks each), shoulder/elbow/wrist of both arms, hand sizes, nose position and hand-visible flags.
- Signer-independent split: 27 / 5 / 11 signers for train / val / test (293 / 84 / 238 clips), so test signers are never seen during training.
- The keypoint dataset (`data.npz`, `signs.json`, `data.meta.csv`) is derived from ASL Citizen and is not redistributed here.

## Method

- ST-GCN backbone from [MMAction2](https://github.com/open-mmlab/mmaction2) (base_channels=120, 10.8M parameters) on a 49-node graph (two hands + upper-body joints), input channels: position, size/visibility flags, and two frame differences.
- Training: AdamW (lr 1e-3), batch size 32, up to 200 epochs, early stopping (patience 30) on validation macro-F1, label smoothing 0.1, head dropout 0.5, gradient clipping 1.0.
- 3 seeds (0, 1, 2); final prediction is the mean of the 3 seeds' softmax outputs (ensemble).
- Metrics: accuracy, macro-F1, top-3 / top-5, Wilson 95% CI, expected calibration error, per-class F1, confusion pairs, OK vs LOW_HAND breakdown.
- BiGRU and Conformer baselines were trained under the same 3-seed protocol in the course project; ST-GCN had the best validation macro-F1.

## Results

| Split | Model | Accuracy | Macro-F1 |
|---|---|---|---|
| Val (84 clips, 5 signers) | ST-GCN, mean of 3 seeds | 0.889 | 0.893 |
| Val (84 clips, 5 signers) | ST-GCN, 3-seed ensemble | 0.893 | 0.893 |
| Test (238 clips, 11 unseen signers) | ST-GCN, mean of 3 seeds | 0.787 | 0.783 |
| Test (238 clips, 11 unseen signers) | ST-GCN, 3-seed ensemble | 0.832 | 0.833 |

Test top-3 / top-5 accuracy of the ensemble: 0.937 / 0.971 (40 errors out of 238 clips). The test split is evaluated once, guarded by a lock file in the notebook.

## How to run

1. Open `notebooks/asl_stgcn_final.ipynb` on Kaggle (GPU) and attach a dataset containing `data.npz`, `signs.json` and `data.meta.csv` with the layout above, or run locally with a GPU.
2. Install dependencies: `pip install -r requirements.txt` (the notebook also installs `mmengine`, `mmcv-lite==2.1.0` and `mmaction2 --no-deps` in cell 3b).
3. Run all cells. Outputs go to `results/`: per-seed checkpoints and histories, `summary.csv`, `val_summary.csv`, `test_metrics.csv`, confusion matrices, per-class F1, error lists, and a TorchScript export (`stgcn_asl20_ensemble.pt`, `stgcn_asl20_single.pt`).

Kaggle notebook (with outputs): https://www.kaggle.com/code/stayhome3977/asl-stgcn-final-thoang
