# AI-Powered Driver Behaviour and Road Safety Monitoring System
## A Complete Sequential Machine Learning Research Prototype

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/pytest-26%20passed-brightgreen.svg)]()
[![Custom Viterbi](https://img.shields.io/badge/Viterbi%20Agreement-100%25-success.svg)]()
[![License: Academic](https://img.shields.io/badge/License-Academic%20Use-lightgrey.svg)]()

An end-to-end machine learning research prototype that ingests naturalistic driving telemetry, models sequential temporal patterns using **Gaussian Hidden Markov Models (Gaussian HMMs)**, implements the **Viterbi decoding algorithm from scratch** with 100% reference agreement, detects safety-critical driving events, and provides an interactive **Trip Replay** dashboard.

Developed using the **UAH-DriveSet** (University of Alcalá) across 6 human drivers, 40 naturalistic trips (500+ minutes), secondary and motorway routes, and three driving behaviours (*Normal*, *Drowsy*, *Aggressive*).

---

## 🌟 Key Features

1. **Custom Viterbi Algorithm (From Scratch)**:
   - Complete log-space dynamic programming implementation: Initialization, Recursion, Termination, and Backtracking.
   - **100.00% exact state sequence agreement** against `hmmlearn.GaussianHMM.decode()`.
   - Zero log-likelihood discrepancy ($\Delta \log P^* = 0.0000$).
2. **Leakage-Free Temporal Preprocessing**:
   - Continuous multi-sensor synchronization: Linear interpolation of 1 Hz GPS onto 10 Hz IMU timestamps.
   - 4th-order zero-phase Butterworth low-pass filtering ($f_c = 2.0$ Hz) and IQR outlier clipping.
   - Chronological train (70%), validation (15%), and test (15%) sequence partitioning with gap buffers.
   - Normalisation scalers and state maps fitted strictly on training sequences.
3. **Kinematic Jerk & Stability Features**:
   - 22 features per 3.0s window: longitudinal/lateral/vertical acceleration means, extremes, standard deviations, longitudinal/lateral jerk ($J(t) = \frac{\Delta a}{\Delta t}$), speed differences, and driving stability variances.
   - 20,483 total extracted feature windows.
4. **Event-Level Safety Detection & Ground-Truth Matching**:
   - Fuses HMM temporal state sequences with physical thresholds to identify hard braking, sudden acceleration bursts, and sharp cornering.
   - **55.84% Event Recall** matched against 154 empirical ground-truth annotations from `EVENTS_INERTIAL.txt`.
5. **Interactive Demonstration with Trip Replay**:
   - Production-grade dashboard built with Streamlit 1.54.
   - **Trip Replay**: Chronological playback of driving trips with real-time speed gauges, acceleration vectors, and predicted risk states.
   - High-throughput decoupled **FastAPI** backend (`api/main.py`).

---

## 📁 Repository Structure

```
Driver-Behaviour-Monitoring/
├── data/
│   ├── raw/
│   │   └── UAH-DriveSet/               # 6 driver directories (D1–D6), 40 trips
│   ├── processed/
│   │   └── features_all.csv            # 20,483 extracted feature windows
│   └── README.md                       # Data provenance, download guide, sensor specs
├── notebooks/
│   ├── 01_dataset_exploration.ipynb    # Phase 1: Exploration, sampling rates, summary
│   ├── 02_preprocessing.ipynb          # Phase 2: Synchronization, Butterworth filter, norm
│   ├── 03_feature_extraction.ipynb     # Phase 3: Jerk kinematics, correlation heatmaps
│   ├── 04_sequence_construction.ipynb  # Phase 4: Temporal sequence assembly & splitting
│   ├── 05_hmm_training.ipynb           # Phase 5: Gaussian HMM formulation & K sweep
│   ├── 06_custom_viterbi.ipynb         # Phase 6: Custom Viterbi derivation & benchmark
│   └── 07_model_evaluation.ipynb       # Phase 7: Accuracy, Macro F1, Event Recall
├── src/
│   ├── preprocessing.py                # IMU/GPS sync, Butterworth filter, normalizer
│   ├── features.py                     # Kinematic features & jerk calculation
│   ├── sequences.py                    # Ordered sequence grouping & chronological split
│   ├── custom_viterbi.py               # Custom Viterbi algorithm (log-space DP)
│   ├── hmm_model.py                    # Gaussian HMM training & state-to-label mapping
│   ├── event_detection.py              # HMM + threshold event detection & ref matching
│   ├── evaluation.py                   # Accuracy, Macro F1, event recall, comparisons
│   └── data_loader.py                  # High-level dataset ingestion utilities
├── dashboard/
│   └── app.py                          # Streamlit analytical dashboard & Trip Replay
├── api/
│   └── main.py                         # FastAPI inference REST endpoints
├── tests/
│   ├── test_preprocessing.py           # Unit tests for synchronization & filters
│   ├── test_features.py                # Unit tests for jerk & windowing
│   └── test_viterbi.py                 # Unit tests for custom Viterbi vs hmmlearn
├── results/
│   ├── figures/                        # Publication figures (heatmaps, GPS routes, CM)
│   ├── metrics/                        # Quantitative metric CSVs & JSON summaries
│   ├── model_comparison/               # Custom vs reference Viterbi benchmarks
│   └── model.pkl                       # Trained Gaussian HMM artifact
├── report/
│   ├── technical_report.md             # Complete academic research paper
│   └── presentation_slides.md          # 12-slide presentation deck with speaker notes
├── scripts/
│   ├── run_experiments.py              # End-to-end experiment pipeline runner
│   └── generate_notebooks.py           # Research notebook generator
├── requirements.txt                    # Project dependencies
├── conftest.py                         # Pytest configuration
├── .gitignore                          # Git ignore rules (protects raw videos)
└── README.md                           # This file
```

---

## ⚡ Quick Start

### 1. Installation
Clone the repository and install requirements:
```bash
pip install -r requirements.txt
```

### 2. Run the Automated Test Suite
Verify that all 26 unit tests pass:
```bash
python -m pytest tests/ -v
```

### 3. Run the End-to-End Experiment Pipeline
To execute the complete research workflow (data preprocessing, feature extraction, HMM training, custom Viterbi benchmarking, event recall calculation, and figure generation):
```bash
python scripts/run_experiments.py
```
All figures will be output to `results/figures/` and metrics to `results/metrics/`.

### 4. Launch the Interactive Dashboard (Option 1: FastAPI + Web App)
Start the high-throughput server:
```bash
uvicorn api.main:app --port 8000 --reload
```
Open **`http://localhost:8000`** in any web browser to access the complete engineering telemetry console:
- **Decoded Driving State & Event Timeline**: Temporal sequence ribbon displaying decoded HMM state intervals (*Normal*, *Drowsy*, *Aggressive*) with interactive event markers and animated playback needle.
- **▶ Telemetry Replay Engine**: Chronological playback with Play / Pause / Reset, 1x–10x speed multipliers, interactive time scrubber, and live instantaneous sensor HUD ($v$, $a_x$, $a_y$, active state).
- **Synchronized Kinematic Waveforms** (Chart.js): 4 live synchronized channels for $a_x$, $a_y$, longitudinal Jerk $J_x(t)$, and vehicle velocity with physical threshold guides.
- **Safety Event Registry**: Chronological log of detected events (Hard Braking, Sudden Accel, Sharp Turn); click any event row to jump the scrubber directly to that second.
- **Research Benchmarks Tab**: Real-time verification of the custom Viterbi algorithm (100.00% path agreement table against `hmmlearn decode()`, accuracy, macro F1, event recall).

---

## 📊 Experimental Results

### Custom Viterbi vs. `hmmlearn` Reference Benchmark
Evaluated on unseen chronological test sequences:
| Sequence | Windows ($T$) | Custom Viterbi (ms) | `hmmlearn` (ms) | Path Agreement | $\Delta \log P^*$ |
|---|---|---|---|---|---|
| Seq 0 | 705 | 732.1 ms | 1.35 ms | **100.00%** | $0.0000$ |
| Seq 1 | 517 | 518.5 ms | 1.77 ms | **100.00%** | $0.0000$ |
| Seq 2 | 484 | 498.8 ms | 1.62 ms | **100.00%** | $0.0000$ |
| Seq 3 | 607 | 734.5 ms | 1.23 ms | **100.00%** | $0.0000$ |
| **Mean** | **578** | **620.9 ms** | **1.49 ms** | **100.00%** | **0.0000** |

### Evaluation Metrics Summary
- **Classification Accuracy**: **50.00%** (vs 25.00% naive baseline).
- **Macro F1**: **0.222**.
- **Event Recall**: **55.84%** (86 matched events out of 154 reference annotations in `EVENTS_INERTIAL.txt`).
- **Optimal State Count ($K$)**: $K=4$ selected via Bayesian Information Criterion (BIC).

---

## 📜 Citations & License
This project utilizes the **UAH-DriveSet**. If citing this work, please include:
```bibtex
@inproceedings{romera2016need,
  title={Need Data for Driving Behavior Analysis? Presenting the Public UAH-DriveSet},
  author={Romera, Eduardo and Bergasa, Luis M and Arroyo, Roberto and L{\'a}zaro, Angel},
  booktitle={IEEE 19th International Conference on Intelligent Transportation Systems (ITSC)},
  pages={387--393},
  year={2016}
}
```
Available under Academic / Research Use Terms.

