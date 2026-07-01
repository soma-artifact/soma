# SOMA Architecture — Paper-to-Code Mapping

## Pipeline Overview

```
                        ┌─────────────────────────────────────┐
                        │     PRE-TRAINING DIAGNOSTIC          │
                        │     sr_computation/                  │
                        │     pid_decomposition.py             │
                        │     → Synergy Ratio (SR)             │
                        │     → Architecture recommendation    │
                        └──────────────┬──────────────────────┘
                                       │
                           SR < 0.05? ───┤──── SR ≥ 0.15?
                           Use SOMA      │     Use XGBoost
                                       │
                         ┌──────────────▼──────────────────────┐
                         │     SOMA CLASSIFIER                  │
                         │     soma_classifier/bilevel_sgd.py  │
                         └──────────────┬──────────────────────┘
                                       │
            ┌───────────────────────────┼───────────────────────────┐
            │                           │                           │
     ┌──────▼──────┐            ┌───────▼──────┐           ┌───────▼──────┐
     │  Group 1    │            │  Group 2     │           │  Group 3     │
     │  SGD Model  │            │  SGD Model   │           │  SGD Model   │
     │  → p₁, H₁  │            │  → p₂, H₂   │           │  → p₃, H₃   │
     └──────┬──────┘            └───────┬──────┘           └───────┬──────┘
            │                           │                           │
            └───────────────────────────┼───────────────────────────┘
                                       │
                         soma_classifier/entropy_features.py
                         [p₁, H₁, p₂, H₂, p₃, H₃, KL₁₂, KL₁₃, KL₂₃, ΔH₁₂, ΔH₁₃, ΔH₂₃]
                                       │
                                 ┌──────▼──────┐
                                 │ Meta-SGD    │
                                 │ (Outer)     │
                                 └──────┬──────┘
                                       │
                               Prediction + Attribution
```

## Paper Section → Source File Mapping

| Paper Section | Key Concept | Source File | Key Function/Class |
|---|---|---|---|
| §2 Background | PID theory | `sr_computation/pid_decomposition.py` | `compute_synergy_diagnostic()` |
| §3.1 SR Definition (Eq. 2) | Synergy Ratio | `sr_computation/pid_decomposition.py` | `SynergyDiagnostic.synergy_ratio` |
| §3.2 Formal Properties | Propositions 1-3 | `docs/theorem.md` | Mathematical proofs |
| §4.1 Semantic Groups | Group definitions | `datasets/*/loader.py` | `GROUPS` constant |
| §4.2 Inner Layer (Eq. 3) | Per-group SGD | `soma_classifier/bilevel_sgd.py` | `_train_inner_models()` |
| §4.3 12D Vector (Eq. 4) | Meta-features | `soma_classifier/entropy_features.py` | `build_12d_meta_vector()` |
| §4.4 Outer Layer (Eq. 5) | Meta-classifier | `soma_classifier/bilevel_sgd.py` | `evaluate_bilevel_sgd()` |
| §5.1 SR Diagnostic | Table 2 | `scripts/run_all.py` | `run_dataset_experiment()` |
| §5.2 Classification | Table 3 | `scripts/run_all.py` | `evaluate_baselines()` |
| §5.3 Ablation | Table 4 | `scripts/run_all.py` | `use_entropy=False` |
| §5 NASA PROMISE | Table 2 (bottom) | `scripts/run_full_experiments.py` | `main()` |
| Figures 1-5 | Publication figs | `scripts/generate_figures.py` | Top-level script |

## Mathematical Formulas → Code

| Formula | Paper Ref | Code |
|---|---|---|
| H(p) = −p·log₂(p) − (1−p)·log₂(1−p) | Eq. in §2.2 | `entropy_features.py → binary_entropy()` |
| KL(p‖q) = p·log₂(p/q) + (1−p)·log₂((1−p)/(1−q)) | Eq. in §2.2 | `entropy_features.py → kl_divergence()` |
| SR = Σ Syn(Xi,Xj;Y) / Σ I(Xk;Y) | Eq. 2 | `pid_decomposition.py → SynergyDiagnostic` |
| J = TPR − FPR | §4.4 | `bilevel_sgd.py → _youdens_j()` |
