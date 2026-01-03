# Modality Balancing (MM'23) in MMRec

This document describes how Modality Balancing is integrated into `gmXian/MMRec` and how to reproduce the paper's defense/attack settings.

## 1) Repository integration points (code mapping)

### Model registry / factory
- Model instantiation is string-based via `src/utils/utils.py` → `get_model()` which imports `models.<model_name_lower>` and then fetches the class by name.
- Entry point: `src/main.py` uses `quick_start()` to load model/config.

### Training loop and loss
- Training loop is in `src/common/trainer.py`:
  - `_train_epoch()` iterates batches and calls `model.calculate_loss(interaction)`.
  - The Modality Balancing loss is added in the same place before backprop (`Trainer._train_epoch`, using `ModalityBalancing.apply_modality_balancing_step`).

### Data loading and BPR triplets
- `src/utils/dataloader.py` (`TrainDataLoader._get_neg_sample`) constructs `(u, i, j)` triples in a tensor shaped `[3, batch_size]`.
- Evaluation data uses `EvalDataLoader` for full-sort metrics.

### Multi-modal features
- `src/common/abstract_recommender.py` loads raw features (`v_feat`, `t_feat`) on the GPU (if `is_multimodal_model=True` and `end2end=False`).
- These are stored as tensors in each model instance and used by modality encoders.

### Modality encoders / embeddings (per model)
- **VBPR** (`src/models/vbpr.py`): item features are fed through linear projection; defense adds per-modality projection heads for embeddings used in Modality Balancing.
- **MMGCN** (`src/models/mmgcn.py`): modality encoders are `GCN` blocks; defense re-runs per-modality GCN for embeddings.
- **GRCN** (`src/models/grcn.py`): modality encoders are `CGCN`; defense uses per-modality CGCN outputs.
- **SLMRec** (`src/models/slmrec.py`): modality embeddings are computed via a LightGCN-style graph over dense features; defense reuses these per-modality embeddings.

Supported models for Modality Balancing in this implementation: **VBPR**, **MMGCN**, **GRCN**, **SLMRec** (all available in this repo). Other models can be supported by implementing the same hooks used by the defense module.

### Scoring function
- All supported models expose dot-product scoring on user/item embeddings; for defense, single-modality margins are computed via dot products over per-modality embeddings.

## 2) Implemented Modality Balancing module

- Module: `src/common/defense/modality_balancing.py`
- Main APIs:
  - `distill_adv_embedding(model, modality, pos_item, neg_item)` uses PGD with L∞ projection to create adversarial embeddings in feature space.
  - `compute_balance_loss(sensitive, robust, u, i, j)` computes the balance loss:
    - `s_margin^t = <e_u^t, e_i^t> - <e_u^t, e_j^t>`
    - `s_margin^v = <e_u^v, e_i^v> - <e_u^v, e_i_adv^v>`
    - `L_balance = max(s_margin^v - s_margin^t, 0)`
  - `apply_modality_balancing_step(interaction)` samples `N` triples from the batch, computes `L_balance`, and adds it to `L_BPR`.

Required model hooks (implemented in the models listed above):
- `get_modalities()` → available modalities (`visual`, `textual`, ...)\n- `get_modal_features(modality)` / `set_modal_features(modality, features)`\n- `get_user_item_embeddings(modality, user_ids, item_ids)`\n- `encode_item_feature(modality, item_ids, feature_override=None)`\n- `get_fused_embeddings()` and `prepare_full_sort()` (for attack evaluation)

## 3) Attack evaluation (test-time)

- Attack mode is FGSM with L2-normalized gradient direction:
  - `Δ_m = ε_m * Γ_m / ||Γ_m||_2`, `Γ_m = ∂L_BPR / ∂Δ_m`
  - `ε_m = eps_ratio * ||x_m||` (default 5% of each item’s feature norm).
- Attack implementation lives in `src/common/defense/modality_balancing.py` and is invoked from `Trainer.evaluate()` when `attack.enable=true`.

## 4) Configuration / usage

### Default configuration (in `src/configs/overall.yaml`)
```
defense:
  enable: False
  method: "modality_balancing"
  sensitive_modality: "visual"
  robust_modality: "textual"
  lambda: 0.01
  N: 20
  distill_eps: 1.0
  pgd_steps: 10
  pgd_step_size: 0.1

attack:
  enable: False
  eps_ratio: 0.05
  mode: "fgsm_l2"
  sample_size: 1024
```

### Example configs
- `src/configs/baby_grcn_modbal.yaml`
  - sensitive: visual, robust: textual
- `src/configs/clothing_grcn_modbal.yaml`
  - sensitive: textual, robust: visual

### Example commands
From `src/`:
- Train with Modality Balancing:
```
python main.py --model GRCN --dataset baby --config_file configs/baby_grcn_modbal.yaml
```
- Evaluate with attack (clean + attack metrics):
```
python main.py --model GRCN --dataset baby --config_file configs/baby_grcn_modbal.yaml --eval_attack
```

> Note: `--eval_attack` is equivalent to setting `attack.enable=true` in the config.

## 5) Sanity checks

- With `defense.enable=false`, training reproduces baseline behavior (clean metrics are unchanged).
- With `defense.enable=true`, clean metrics remain stable (small or no regression), while attack metrics improve.
