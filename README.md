# SOMA: Entropy-Calibrated Multi-View Failure Prediction for Predictive Maintenance

This codebase implements the Synergy Ratio (SR) diagnostic derived from Partial Information Decomposition (PID) and the Synergy Oriented Model Assessment (SOMA) classifier for data-driven model architecture selection. The framework allows practitioners to assess whether a failure prediction domain contains significant cross-group synergy before selecting a model architecture, helping to choose between simpler interpretable models or complex ensembles. The implementation is validated on eight industrial failure prediction and software defect datasets.

---

## 1. Repository Structure

The code is organized as follows:

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
│   ├── approach.md             #   Design approach documentation
│   ├── baseline_comparison.md  #   Baseline comparison documentation
│   ├── architecture.md         #   Detailed section-to-code maps
│   ├── mathematical_explanation.md # Step-by-step algebraic breakdown of SOMA features
│   ├── technical_explanation.md#   Semantic groupings and pipeline explanations
│   └── extending.md            #   Guide to adding new datasets/models
```

---

## 2. Setup Instructions

The repository relies on standard scientific Python libraries. A preconfigured virtual environment setup script is provided.

### Prerequisites

- Python 3.10 or higher
- GCC (required to compile the information theory package `dit`)

### Virtual Environment Setup

To create the virtual environment and install all dependencies:

```bash
chmod +x setup.sh
./setup.sh
```

Alternatively, to set up the environment manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Verifying the Setup

Verify that all modules can import correctly and the environment is operational by running the test execution coordinator:

```bash
source .venv/bin/activate
python3 reproduce_all.py
```

This sequentially runs a full verification pass, which executes all experiments, verifies mathematical consistency, profile resource usage, and regenerates publication-ready figures.

---

## 3. Reproducing Paper Results

The table below maps each paper table and figure to the exact script command that reproduces it:

| Paper Element | Description | Execution Command |
|---|---|---|
| Table I | Classification Performance (AUC, MCC, F1, Brier) | `python scripts/run_soma_evaluation.py` |
| Table II | Synergy Ratio (SR) Diagnostic and 95% Confidence Intervals | `python scripts/run_sr_diagnostic.py` (Primary) <br> `python scripts/run_full_experiments.py` (Promise) |
| Table III | Synergy Ratio Estimator Consistency Check (BROJA, Imin, CoI) | `python experiments/multi_estimator_sr.py` |
| Table IV | Feature Ablation study (Full 12D vs 3D) and Paired t-Test | `python scripts/run_ablation.py` |
| Figure 1 | Distribution of PID Information Atoms per dataset | `python scripts/generate_figures.py` |
| Figure 2 | Synergy Ratio vs SOMA to XGBoost Performance Gap | `python scripts/generate_figures.py` |
| Figure 3 | AUC Model Comparison across datasets | `python scripts/generate_figures.py` |
| Figure 4 | Paired t-test Ablation Results | `python scripts/generate_figures.py` |
| Figure 5 | Synergy Ratio Thresholds Overview | `python scripts/generate_figures.py` |

To run the entire pipeline at once, execute the master shell script:

```bash
./reproduce_all.sh
```

---

## 4. File-by-File Reference

| File Path | Purpose | Paper Reference |
|---|---|---|
| `sr_computation/pid_decomposition.py` | Implements Partial Information Decomposition (PID), Williams-Beer $I_{\min}$, and Synergy Ratio (SR) bootstraps | Section III (Equations 1-5), Algorithm 1 |
| `soma_classifier/entropy_features.py` | Quantile discretization and 12D entropy-KL meta-feature vector construction | Section IV (Equation 6) |
| `soma_classifier/bilevel_sgd.py` | Specialist training (inner SGDs) and Generalist Meta-Classifier (outer SGD) | Section IV (Algorithms 2 and 3) |
| `datasets/promise_loader_base.py` | Shared utility base class for loading software defect metrics | Section VI-E |
| `experiments/multi_estimator_sr.py` | Verifies SR stability across different mutual information estimators | Section V-D (Table III) |
| `experiments/run_surprisingness_sr.py` | Computes surprisingness-weighted Synergy Ratio corrections | Section V-C |
| `experiments/benchmark_efficiency.py` | Measures execution speed, model parameter sizes, and memory usage | Section VI-D, Section V-E |

---

## 5. Datasets

The repository includes a diverse set of industrial predictive maintenance and software defect datasets:

1. **AI4I 2020 Predictive Maintenance Dataset (`datasets/ai4i/`)**
   - *Source:* UCI Machine Learning Repository (ID 601).
   - *Structure:* 10,000 tool logs, grouped into 3 telemetry groups (Thermal, Mechanical, Wear).
   - *Preprocessing:* Continuous columns are quantile-discretized into $B=8$ bins for diagnostic computation.

2. **NASA C-MAPSS Turbofan Degradation Dataset (`datasets/cmapss/`)**
   - *Source:* NASA Prognostics Data Repository (FD001).
   - *Structure:* 25,759 sensor readings grouped into 3 groups (Temperature, Pressure, Speed).
   - *Preprocessing:* Features normalized; remaining Useful Life (RUL) thresholded to binary failure labels.
   - *Note:* Due to file size restrictions, the repository contains a compressed subset for immediate verification. The complete public dataset can be downloaded from the NASA Prognostics Repository.

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

## 6. Hyperparameter Reference

The table below lists every configuration value used in SOMA, its setting, and the paper section that specifies it:

| Hyperparameter | Value | Description | Paper Section |
|---|---|---|---|
| Discretization Bins ($B$) | 8 | Number of quantile bins for continuous feature discretization | Section III, Section IV |
| Subsample Size ($N$) | 2000 | Number of rows sampled for Partial Information Decomposition | Section III |
| Bootstrap Samples | 50 / 100 | Bootstrap iterations for computing 95% confidence intervals | Section V-D |
| Confidence Level | 0.95 | Significance level for the Synergy Ratio confidence interval | Section III-C |
| CV Folds | 10 | Stratified cross-validation splits for classification evaluation | Section VI-B |
| Inner SGD Penalty | elasticnet | Regularization type for specialist classifiers | Section IV-B |
| Inner SGD L1 Ratio | 0.15 | Mix parameter between L1 and L2 penalty for specialists | Section IV-B |
| Outer SGD Penalty | elasticnet | Regularization type for generalist meta-classifier | Section IV-D |
| Outer SGD L1 Ratio | 0.15 | Mix parameter between L1 and L2 penalty for generalist | Section IV-D |
| Decision Metric | Youden's J | Objective function maximized for decision threshold selection | Section IV-D |

---

## 7. Known Limitations and Reproducibility Notes

- **BROJA/NumPy 2.x Compatibility:** The `dit` package (used for BROJA PID calculations) relies on deprecated NumPy functions (`np.alltrue`, `np.product`) that were removed in NumPy 2.x. This repository injects load-time patches (`np.alltrue = np.all`, etc.) in `sr_computation/pid_decomposition.py` and `experiments/multi_estimator_sr.py` to guarantee seamless compatibility with modern Python environments.
- **Imin Fallback:** If the `dit` package is missing or fails, the pipeline automatically falls back to the Williams-Beer $I_{\min}$ estimator.
- **Bootstrap Subsampling:** Due to the high computational complexity of Partial Information Decomposition, the pipeline standardizes on subsampling $N=2000$ points for the PID/Synergy Ratio calculation. Subsampling prevents CPU freeze-ups during bootstrap iterations while maintaining stable confidence intervals.
- **NASA Promise Imbalance:** On extremely small or sparse datasets (such as CM1), the stratified split combined with SMOTE may exhibit slight variance in MCC metrics compared to monolithic training due to minor random seed variations.

---

## 8. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
