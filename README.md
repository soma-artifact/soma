# SOMA: Entropy-Calibrated Failure Prediction Pipeline

## 1. Overview
This repository implements the Synergy Ratio (SR) information-theoretic diagnostic and the SOMA (Stochastic Specialist Multi-view Aggregator) classification framework. The pipeline provides a formal quantitative methodology for industrial failure and software defect prediction by determining when multi-view sensor inputs exhibit high synergy—warranting complex model fusion architectures—and when low synergy dominates, indicating that simple, interpretable linear models are sufficient. Under the hood, the SOMA framework constructs a 12-dimensional meta-feature space of entropy, Kullback-Leibler (KL) divergence, and class predictions to optimize failure detection.

---

## 2. Repository Structure

The codebase is organized into modular packages to isolate algorithms, datasets, runnable scripts, and experimental results:

```
├── datasets/                   # Self-contained loader packages for each dataset
│   ├── promise_loader_base.py  #   Unified base ARFF loader for NASA Defect benchmarks
│   ├── ai4i/                   #   AI4I 2020 predictive maintenance loader & CSV data
│   ├── cmapss/                 #   NASA C-MAPSS turbofan degradation loader & telemetry
│   ├── smd/                    #   Server Machine Dataset (SMD) loader & metrics
│   ├── synthetic/              #   Synthetic synergy generator and loader
│   ├── cm1/                    #   NASA Software Defect CM1 loader & ARFF data
│   ├── jm1/                    #   NASA Software Defect JM1 loader & ARFF data
│   ├── pc1/                    #   NASA Software Defect PC1 loader & ARFF data
│   └── mc2/                    #   NASA Software Defect MC2 loader & ARFF data
│
├── sr_computation/             # Information-theoretic algorithms
│   └── pid_decomposition.py    #   Partial Information Decomposition (PID) & Synergy Ratio (SR)
│
├── soma_classifier/            # Core SOMA model implementation
│   ├── bilevel_sgd.py          #   Bi-level SGD Specialist experts & Generalist meta-classifier
│   └── entropy_features.py     #   Quantile discretization, entropy, and KL-divergence features
│
├── scripts/                    # Top-level runnable CLI entry points
│   ├── run_sr_diagnostic.py    #   Diagnostic interface to compute SR on any dataset
│   ├── run_soma_evaluation.py  #   SOMA vs baselines evaluation (10-fold CV)
│   ├── run_ablation.py         #   12D vs 3D feature ablation and paired t-test
│   ├── generate_figures.py     #   Matplotlib publication figure generator
│   ├── run_all.py              #   Unified master experiment runner
│   └── run_full_experiments.py #   SOMA pipeline evaluation on NASA defect datasets
│
├── experiments/                # Research analysis and consistency checks
│   ├── multi_estimator_sr.py   #   Estimator consistency comparison (BROJA, Imin, CoI)
│   ├── run_surprisingness_sr.py#   Surprisingness-weighted Synergy Ratio checks
│   └── benchmark_efficiency.py #   Execution time, memory, and model size benchmarks
│
├── results/                    # Generated outputs
│   ├── tables/                 #   JSON tables, metrics, and consistent run dumps
│   └── figures/                #   Auto-generated PNG figures (Figures 1–5)
│
├── docs/                       # Technical and mathematical documentation
│   ├── architecture.md         #   Detailed section-to-code maps
│   ├── mathematical_explanation.md # Step-by-step algebraic breakdown of SOMA features
│   ├── technical_explanation.md#   Semantic groupings and pipeline explanations
│   └── extending.md            #   Guide to adding new datasets/models
│
├── requirements.txt            # Python package dependencies
├── setup.sh                    # Automation script for environment setup
├── reproduce_all.py            # Python execution orchestrator
├── reproduce_all.sh            # Top-level bash execution wrapper
└── LICENSE                     # MIT license file
```

---

## 3. Setup Instructions

The setup is automated and handles all package installations, environment creation, and verification steps.

### Prerequisites
- **Python Version:** 3.9, 3.10, 3.11, or 3.12 installed on your system.
- **Operating System:** Linux, macOS, or Windows (via WSL).

### 1. Build Virtual Environment and Install Dependencies
From the repository root directory, run the setup script:

```bash
chmod +x setup.sh
./setup.sh
```

This script will:
- Initialize an isolated virtual environment (`.venv/`)
- Upgrade `pip` to the latest version
- Install all required libraries listed in `requirements.txt`
- Execute a verification script to test that all package imports (including `dit`, `xgboost`, `scikit-learn`, `scipy`, etc.) load successfully.

### 2. Activate the Environment
Before executing any script, activate the virtual environment:

```bash
source .venv/bin/activate
```

---

## 4. Reproducing Results

To replicate the tables, figures, and statistical analyses reported in the paper, choose one of the following methods:

### Option A: Run the Complete Reproduction Suite (Recommended)
You can run the entire replication pipeline in one step. It executes all experiments, statistical tests, consistency checks, and regenerates all figures:

```bash
chmod +x reproduce_all.sh
./reproduce_all.sh
```

*(This runs a fast cross-validation configuration to complete the entire suite in ~5–8 minutes, outputting all figures to `results/figures/` and metrics to `results/tables/`).*

### Option B: Step-by-Step Table and Figure Replication

#### 1. Table II (Synergy Ratio across Eight Datasets)
To compute the Synergy Ratio (SR) and print the Partial Information Decomposition (PID) atoms for any dataset:
```bash
python scripts/run_sr_diagnostic.py --dataset AI4I
python scripts/run_sr_diagnostic.py --dataset C-MAPSS
python scripts/run_sr_diagnostic.py --dataset SMD
python scripts/run_sr_diagnostic.py --dataset Synthetic
python scripts/run_sr_diagnostic.py --dataset CM1
python scripts/run_sr_diagnostic.py --dataset JM1
python scripts/run_sr_diagnostic.py --dataset PC1
python scripts/run_sr_diagnostic.py --dataset MC2
```

#### 2. Table III (SOMA Specialist AUC-ROC Performance vs Baselines)
To train the SOMA specialist SGD expert models and compare their AUC-ROC, MCC, and F1 metrics against standard machine learning baselines (Naive Bayes, Logistic Regression, Random Forest, XGBoost) using 10-fold cross-validation:
```bash
python scripts/run_soma_evaluation.py
```

#### 3. Table IV (12D vs 3D Feature Ablation and Paired t-Test)
To evaluate the contribution of entropy and KL-divergence meta-features by training a SOMA classifier with (12D) and without (3D) them, running a Relational Paired t-Test over the cross-validation folds:
```bash
python scripts/run_ablation.py
```

#### 4. Figures 1 to 5 (Publication Figures)
To regenerate the publication-ready figures using the experiment results:
```bash
python scripts/generate_figures.py
```
This produces 5 PNG files in `results/figures/`:
- `fig1_pid_decomposition.png`: Distribution of PID information atoms per dataset.
- `fig2_sr_vs_gap.png`: Correlation mapping the Synergy Ratio to SOMA's performance improvement over baseline models.
- `fig3_model_comparison.png`: Consolidated AUC comparison across models.
- `fig4_ablation.png`: Paired t-test results demonstrating the value of entropy features.
- `fig5_synergy_overview.png`: Cross-dataset Synergy Ratio thresholds.

---

## 5. File-by-File Reference

| Path | File Name | Description |
|---|---|---|
| `sr_computation/` | `pid_decomposition.py` | Implementation of PID information atoms and Synergy Ratio (SR) calculation. |
| `soma_classifier/` | `bilevel_sgd.py` | Implementation of SOMA's specialists SGD training and outer Meta-Classifier logic. |
| `soma_classifier/` | `entropy_features.py` | Discretization algorithms and computation of 12-dimensional entropy-KL meta-vectors. |
| `datasets/` | `promise_loader_base.py` | Common utility class for loading NASA Promise software defect ARFF datasets. |
| `scripts/` | `run_sr_diagnostic.py` | CLI tool to compute SR and print PID atom values for a given dataset. |
| `scripts/` | `run_soma_evaluation.py` | Pipeline runner evaluating SOMA against standard baseline classifiers. |
| `scripts/` | `run_ablation.py` | Feature ablation script comparing 12D SOMA to 3D models with t-test. |
| `scripts/` | `generate_figures.py` | Visualizer generating all publication figures based on experiment outputs. |
| `scripts/` | `run_all.py` | Execution coordinator running full experiments for the 4 primary datasets. |
| `scripts/` | `run_full_experiments.py` | Execution coordinator running full experiments for the 4 NASA Promise datasets. |
| `experiments/` | `multi_estimator_sr.py` | Consistency comparison script verifying SR behavior across different PID estimators. |
| `experiments/` | `run_surprisingness_sr.py` | Diagnostic script verifying surprisingness-weighted Synergy Ratio metrics. |
| `experiments/` | `benchmark_efficiency.py` | Performance profiler measuring SOMA training/inference speed, memory, and disk footprint. |

---

## 6. Datasets

The repository includes a diverse set of industrial predictive maintenance and software defect datasets:

1. **AI4I 2020 Predictive Maintenance Dataset (`datasets/ai4i/`)**
   - *Source:* UCI Machine Learning Repository.
   - *Structure:* 10,000 tool logs, grouped into 3 distinct telemetry groups (Thermal, Mechanical, Wear).
   - *Preprocessing:* Continuous columns are quantile-discretized into $B=8$ bins for diagnostic computation.

2. **NASA C-MAPSS Turbofan Degradation Dataset (`datasets/cmapss/`)**
   - *Source:* NASA Prognostics Data Repository (FD001).
   - *Structure:* 25,759 sensor readings grouped into 3 groups (Temperature, Pressure, Speed).
   - *Preprocessing:* Features normalized; remaining Useful Life (RUL) thresholded to binary failure labels.

3. **Server Machine Dataset (SMD) (`datasets/smd/`)**
   - *Source:* Public system monitoring logs from a large internet company.
   - *Structure:* Telemetry timeseries split into Compute, Memory, and Network groups.
   - *Preprocessing:* Downsampled to 9,950 points with sliding anomaly labels.

4. **Synthetic Cascading Synergy Dataset (`datasets/synthetic/`)**
   - *Source:* Generative simulator included in the repository.
   - *Structure:* Simulates a multi-layered software pipeline (Broker, Consumer, Network) with cascading failure delays.

5. **NASA Software Defect Datasets (`datasets/cm1/`, `datasets/jm1/`, `datasets/pc1/`, `datasets/mc2/`)**
   - *Source:* Open Science Promise Repository.
   - *Structure:* Code metrics grouped into Halstead, Complexity, and McCabe code volumes.
   - *Preprocessing:* Null entries and duplicate rows removed; labels represent defective functions.

---

## 7. Known Limitations & Implementation Notes

- **BROJA/NumPy 2.x Compatibility:** The `dit` package (used for BROJA PID calculations) uses deprecated NumPy functions (`np.alltrue`, `np.product`) that were removed in NumPy 2.x. This repository injects load-time patches (`np.alltrue = np.all`, etc.) in `sr_computation/pid_decomposition.py` and `experiments/multi_estimator_sr.py` to guarantee seamless compatibility with modern Python environments.
- **Imin Fallback:** If the `dit` package is missing or fails, the pipeline automatically falls back to the Williams-Beer $I_{\min}$ estimator.
- **Bootstrap Subsampling:** Due to the high computational complexity of Partial Information Decomposition, the pipeline standardizes on subsampling $N=2000$ points for the PID/Synergy Ratio calculation. Subsampling prevents CPU freeze-ups during bootstrap iterations while maintaining stable confidence intervals.

---

## 8. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
