# MAPTab: Missing-Aware Progressive Masking for Self-Supervised Tabular Imputation
Missing values are pervasive in tabular data, compromising data quality and reliability. We present MAPTab, a self-supervised imputation frame-work that improves masked modeling for incomplete heterogeneous tables. Existing methods construct training objectives via random masking, which fails to capture non-uniform missingness patterns, and use fixed masking intensity, which cannot simultaneously ensure stable early optimization and strong late-stage recovery. MAPTab addresses these limitations with missing-aware progressive masking, which performs weighted re-masking according to each field’s empirical missing tendency and gradually increas-es masking intensity, yielding better task alignment and robust recovery through easy-to-hard learning. MAPTab further introduces imputation-oriented tabular tokenization, which jointly encodes feature values, field identity, and missingness status, and a type-aware reconstruction objective that separately models continuous and categorical variables within a uni-fied Transformer framework. Experiments on nine public datasets show that MAPTab consistently outperforms 15 competitive baselines across three missingness mechanisms and multiple evaluation settings, achieving the best average rank of 1.3.


## Design Motivation
![Design rationale of MAPTab.](./figures/motivation.png)

The left panel summarizes three core challenges in imputation-oriented masked modeling for incomplete tabular data, while the right panel illustrates the three key components of MAPTab: missing-aware progressive masking, imputation-oriented tabular tokenization, and type-aware reconstruction.


## Overall Architecture of MAPTab
![Overall architecture of MAPTab.](./figures/model.png)

Given an incomplete tabular sample, MAPTab first applies missing-aware progressive masking to construct self-supervised reconstruction targets. The partially observed input is then converted into structured tokens that jointly encode feature values, field identity, and missingness status for Transformer-based context modeling. A type-aware decoder finally reconstructs the full field sequence, using separate prediction heads for continuous and categorical variables to impute naturally missing entries.

We implemented this architecture using **PyTorch**.


## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
python main.py --config src/config/default.yaml
```
