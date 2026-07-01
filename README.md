# SOMA: Entropy-Calibrated Multi-View Failure Prediction for Predictive Maintenance

This codebase implements the Synergy Ratio (SR) diagnostic derived from Partial Information Decomposition (PID) and the Synergy Oriented Model Assessment (SOMA) classifier for data-driven model architecture selection. The framework allows practitioners to assess whether a failure prediction domain contains significant cross-group synergy before selecting a model architecture, helping to choose between simpler interpretable models or complex ensembles. The implementation is validated on eight industrial failure prediction and software defect datasets.

---

## 1. System Requirements

- **Supported Operating Systems**:
  - **Linux**: Tested on Ubuntu 22.04 LTS (x86_64)
  - **macOS**: Tested on macOS 13 (Ventura) / 14 (Sonoma) on Apple Silicon and Intel architectures
  - **Windows**: Tested on Windows 10 / 11 (x86_64)
- **Minimum RAM**: 8 GB RAM (16 GB recommended for high-dimensional bootstrap routines)
- **Disk Space**: Approximately 100 MB free space
- **Prerequisites (C Compiler Setup)**:
  - Compiling C extensions for the `dit` information theory library requires a C compiler:
    - **Linux**: Install GCC (e.g. `sudo apt-get install build-essential python3-dev`)
    - **macOS**: Install Xcode Command Line Tools (run `xcode-select --install` in terminal)
    - **Windows**: Install Microsoft Visual C++ 14.0 or greater (available via Visual Studio Build Tools)
  - **Python**: Version 3.10 or higher installed and added to the system path

---

## 2. Cloning the Repository

To clone this repository and enter the workspace directory, run in your terminal:

```bash
git clone https://github.com/soma-artifact/soma.git
cd soma
```

---

## 3. Repository Structure

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
│
├── setup.sh                    # Preconfigured virtual environment setup shell script (Linux/macOS)
├── reproduce_all.sh            # Top-level bash runner executing complete replication suite (Linux/macOS)
├── reproduce_all.py            # Master Python orchestrator running full execution pipeline (All OS)
├── requirements.txt            # Python package dependency definitions
└── LICENSE                     # Project License
```

---

## 4. Setup Instructions

The repository relies on standard scientific Python libraries. Preconfigured workspace setup tools are provided for all environments.

### Virtual Environment Setup

#### On Linux & macOS
To create the virtual environment and install all dependencies automatically:
```bash
chmod +x setup.sh
./setup.sh
```

#### On Windows (PowerShell)
To initialize the environment manually:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### On Windows (Command Prompt)
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Setup Verification Step

Before running any experiments, verify that the virtual environment is correctly built and dependencies compile without issues:

- **Linux & macOS**:
  ```bash
  source .venv/bin/activate
  python3 -c "import dit, sklearn, numpy, scipy; print('Environment OK')"
  ```
- **Windows**:
  ```cmd
  python -c "import dit, sklearn, numpy, scipy; print('Environment OK')"
  ```

If it prints `Environment OK`, all dependencies (including compiled C extensions for `dit`) are ready.

---

## 5. Quick Verification Run

To verify the pipeline execution on a lightweight task before running full benchmarks, execute the diagnostic utility on the synthetic dataset:

- **Linux & macOS**:
  ```bash
  source .venv/bin/activate
  python scripts/run_sr_diagnostic.py --dataset synthetic
  ```
- **Windows (PowerShell)**:
  ```powershell
  python scripts\run_sr_diagnostic.py --dataset synthetic
  ```

- **Expected Output**: The script prints the Synergy Ratio value, 95% Confidence Intervals, and PID atom breakdown (redundancy, unique info, synergy) to the console.
- **Expected Runtime**: Under 60 seconds on a standard modern CPU.

---

## 6. Reproducing Paper Results

The table below maps each paper table and figure to the exact script command that reproduces it.

**Dependency Order**: Run the evaluation scripts (e.g. `run_soma_evaluation.py`, `run_full_experiments.py`, `multi_estimator_sr.py`, `run_ablation.py`) first. These scripts save intermediate JSON result files to `results/tables/`. The figure generator script `generate_figures.py` loads these intermediate results to plot the publication-ready graphics. Therefore, running `generate_figures.py` before executing the evaluations will raise file-not-found errors.

| Paper Element | Description | Execution Command (Linux/macOS) | Windows Command | Expected Output | Expected Runtime (8-core CPU) |
|---|---|---|---|---|---|
| Table I | Classification Performance (AUC, MCC, F1, Brier) | `python scripts/run_soma_evaluation.py` | `python scripts\run_soma_evaluation.py` | Outputs metrics table; saves JSON to `results/tables/experiment_results.json` | ~15 minutes |
| Table II (Top) | Synergy Ratio (SR) Diagnostic on Primary Datasets | `python scripts/run_sr_diagnostic.py` | `python scripts\run_sr_diagnostic.py` | Prints SR value, 95% CI, and PID atom breakdown; saves to `results/tables/broja_sr_results.json` | ~3 minutes |
| Table II (Bottom) | Synergy Ratio (SR) Diagnostic on NASA Defect datasets | `python scripts/run_full_experiments.py` | `python scripts\run_full_experiments.py` | Prints SR and SOMA evaluation statistics to console | ~2 minutes |
| Table III | Synergy Ratio Estimator Consistency Check | `python experiments/multi_estimator_sr.py` | `python experiments\multi_estimator_sr.py` | Prints consistency table; saves JSON to `results/tables/multi_estimator_sr.json` | ~15 seconds |
| Table IV | Feature Ablation study and Paired t-Test | `python scripts/run_ablation.py` | `python scripts\run_ablation.py` | Prints paired t-test results and p-values to console | ~2 minutes |
| Figures 1–5 | Generation of all publication-ready PNG figures | `python scripts/generate_figures.py` | `python scripts\generate_figures.py` | Generates 5 PNG files saved under `results/figures/` | ~5 seconds |

### Full Replication Execution

#### On Linux & macOS
To execute the entire verification pipeline at once, run:
```bash
chmod +x reproduce_all.sh
./reproduce_all.sh
```

#### On Windows
To execute the entire verification pipeline at once, run:
```cmd
python reproduce_all.py
```

`reproduce_all.py` is the actual Python execution orchestrator. On Linux/macOS, `reproduce_all.sh` is a shell wrapper that activates the virtual environment `.venv` and then executes `reproduce_all.py`. On Windows, execute `reproduce_all.py` directly within your activated environment. Running the full reproduction executes the entire pipeline, including all setup verifications and experiment runs, in approximately 10 minutes.

---

## 7. File-by-File Reference

| File Path | Purpose | Paper Reference |
|---|---|---|
| `sr_computation/pid_decomposition.py` | Implements Partial Information Decomposition (PID), Williams-Beer $I_{\min}$, and Synergy Ratio (SR) bootstraps | Section III (Equations 1–5), Algorithm 1 |
| `soma_classifier/entropy_features.py` | Quantile discretization and 12D entropy-KL meta-feature vector construction | Section IV (Equation 6) |
| `soma_classifier/bilevel_sgd.py` | Specialist training (inner SGDs) and Generalist Meta-Classifier (outer SGD) | Section IV (Algorithms 2 and 3) |
| `datasets/promise_loader_base.py` | Shared utility base class for loading software defect metrics | Section VI-E |
| `datasets/ai4i/loader.py` | Loads and groups features for the AI4I 2020 predictive maintenance benchmark | Section V-A |
| `datasets/cmapss/loader.py` | Standardizes turbofan degradation telemetry data and computes binary labels | Section V-A |
| `datasets/smd/loader.py` | Normalizes Server Machine Dataset telemetry and partitions groups | Section V-A |
| `datasets/synthetic/loader.py` | Simulates a controllable cascading pipeline (Broker, Consumer, Network) | Section V-A |
| `datasets/cm1/loader.py` | Metric mapping loader for the NASA CM1 defect dataset | Section VI-E |
| `datasets/jm1/loader.py` | Metric mapping loader for the NASA JM1 defect dataset | Section VI-E |
| `datasets/pc1/loader.py` | Metric mapping loader for the NASA PC1 defect dataset | Section VI-E |
| `datasets/mc2/loader.py` | Metric mapping loader for the NASA MC2 defect dataset | Section VI-E |
| `scripts/run_sr_diagnostic.py` | CLI diagnostic interface to run the SR diagnostic on any registered dataset | Section III, Section VI-B |
| `scripts/run_soma_evaluation.py` | Runs SOMA classification benchmarks against standard models | Section VI-B (Table I) |
| `scripts/run_ablation.py` | Conducts SOMA feature ablation and t-test statistics | Section VI-D (Table IV) |
| `scripts/generate_figures.py` | Renders publication-ready matplotlib visual figures | Figures 1 to 5 |
| `scripts/run_all.py` | Execution runner driving the SOMA evaluation loops | Section VI |
| `scripts/run_full_experiments.py` | Runs the SOMA evaluations on the NASA Promise defect datasets | Section VI-E (Table II) |
| `experiments/multi_estimator_sr.py` | Computes the mutual information and synergy ratio under BROJA, Imin, and Co-Info | Section V-D (Table III) |
| `experiments/benchmark_efficiency.py` | Measures execution latency, model file sizes, and memory usage | Section VI-D, Section V-E |

---

## 8. Datasets

The repository includes a diverse set of industrial predictive maintenance and software defect datasets:

1. **AI4I 2020 Predictive Maintenance Dataset (`datasets/ai4i/`)**
   - *Source:* UCI Machine Learning Repository (ID 601).
   - *URL*: [https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset)
   - *Installation*: The CSV file `ai4i_2020.csv` is fully included in the repository at `datasets/ai4i/ai4i_2020.csv`.
   - *Preprocessing:* Continuous columns are quantile-discretized into $B=8$ bins for diagnostic computation.

2. **NASA C-MAPSS Turbofan Degradation Dataset (`datasets/cmapss/`)**
   - *Source:* NASA Prognostics Data Repository (FD001).
   - *URL*: [https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#turbofan](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#turbofan)
   - *Installation*: A synthetic subset is already included in `datasets/cmapss/CMAPSSData/train_FD001.txt`. To run on the full NASA repository dataset: download the C-MAPSS dataset from the URL, unzip it, and place `train_FD001.txt` inside `datasets/cmapss/CMAPSSData/`.
   - *Preprocessing:* Features normalized; remaining Useful Life (RUL) thresholded to binary failure labels.

3. **Server Machine Dataset (SMD) (`datasets/smd/`)**
   - *Source:* NetManAIOps ServerMachineDataset.
   - *URL*: [https://github.com/NetManAIOps/Anomaly-Decct/tree/master/ServerMachineDataset](https://github.com/NetManAIOps/Anomaly-Decct/tree/master/ServerMachineDataset)
   - *Installation*: The sliding anomaly subsets are fully preprocessed and pre-loaded in `datasets/smd/SMD/`. No additional download is required for replication.
   - *Preprocessing:* Downsampled to 9,950 points with sliding anomaly labels.

4. **Synthetic Cascading Synergy Dataset (`datasets/synthetic/`)**
   - *Source:* Generative simulator included in the repository.
   - *Installation*: Auto-generated at runtime; no download required.
   - *Preprocessing*: Simulates a multi-layered software pipeline (Broker, Consumer, Network) with cascading failure delays.

5. **NASA Software Defect Datasets (`datasets/cm1/`, `datasets/jm1/`, `datasets/pc1/`, `datasets/mc2/`)**
   - *Source:* Open Science Promise Repository.
   - *URL*: [http://promise.site.uottawa.ca/SERepository/](http://promise.site.uottawa.ca/SERepository/)
   - *Installation*: The ARFF files (`CM1.arff`, `JM1.arff`, `PC1.arff`, `MC2.arff`) are fully included in the repository inside their respective dataset folders. No additional download is required.
   - *Preprocessing:* Null entries and duplicate rows removed; labels represent defective functions.

---

## 9. Hyperparameter Reference

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

## 10. Known Limitations and Reproducibility Notes

- **BROJA/NumPy 2.x Compatibility:** The `dit` package (used for BROJA PID calculations) relies on deprecated NumPy functions (`np.alltrue`, `np.product`) that were removed in NumPy 2.x. This repository injects load-time patches (`np.alltrue = np.all`, etc.) in `sr_computation/pid_decomposition.py` and `experiments/multi_estimator_sr.py` to guarantee seamless compatibility with modern Python environments.
- **Imin Fallback:** If the `dit` package is missing or fails, the pipeline automatically falls back to the Williams-Beer $I_{\min}$ estimator.
- **Bootstrap Subsampling:** Due to the high computational complexity of Partial Information Decomposition, the pipeline standardizes on subsampling $N=2000$ points for the PID/Synergy Ratio calculation. Subsampling prevents CPU freeze-ups during bootstrap iterations while maintaining stable confidence intervals.
- **NASA Promise Imbalance:** On extremely small or sparse datasets (such as CM1), the stratified split combined with SMOTE may exhibit slight variance in MCC metrics compared to monolithic training due to minor random seed variations.

---

## 11. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
