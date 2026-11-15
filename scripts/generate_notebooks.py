"""
scripts/generate_notebooks.py — Generates All 7 Reproducible Research Notebooks
================================================================================
Generates:
  01_dataset_exploration.ipynb
  02_preprocessing.ipynb
  03_feature_extraction.ipynb
  04_sequence_construction.ipynb
  05_hmm_training.ipynb
  06_custom_viterbi.ipynb
  07_model_evaluation.ipynb
"""

import json
from pathlib import Path

NOTEBOOKS_DIR = Path("notebooks")
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.13"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }


def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip().splitlines(keepends=True)
    }


def code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code.strip().splitlines(keepends=True)
    }


# ===========================================================================
# 01_dataset_exploration.ipynb
# ===========================================================================
nb01 = make_notebook([
    md_cell("""# Phase 1: UAH-DriveSet Exploration and Understanding

**Project**: AI-Powered Driver Behaviour and Road Safety Monitoring System  
**Objective**: Understand the data provided by the UAH-DriveSet (University of Alcalá, Spain), including sampling frequencies, driver behaviors, road types, and inertial event annotations.
"""),
    code_cell("""import sys
sys.path.insert(0, '..')
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocessing import load_all_trips
from src.data_loader import dataset_summary, label_distribution, driver_summary

sns.set_theme(style="whitegrid")
%matplotlib inline
"""),
    md_cell("""## 1. Load All Trips Across Drivers (D1 – D6)"""),
    code_cell("""dataset_root = Path('../data/raw/UAH-DriveSet')
trips = load_all_trips(dataset_root)
print(f"Total trips loaded: {len(trips)}")
"""),
    md_cell("""## 2. Dataset Structure and Summary Table"""),
    code_cell("""summary_df = dataset_summary(trips)
display(summary_df.head(10))
print(f"Total recorded driving time: {summary_df['duration_s'].sum()/60:.1f} minutes")
"""),
    md_cell("""## 3. Class Balance and Driver Distribution"""),
    code_cell("""fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

lbl_df = label_distribution(trips)
axes[0].bar(lbl_df.index, lbl_df['count'], color=['#2ecc71', '#f39c12', '#e74c3c'], edgecolor='black')
axes[0].set_title("Driving Behavior Distribution", fontweight="bold")
axes[0].set_ylabel("Number of Trips")

drv_df = driver_summary(trips)
drv_df.plot(kind='bar', stacked=True, ax=axes[1], colormap='viridis', edgecolor='black')
axes[1].set_title("Per-Driver Behavioral Profiles", fontweight="bold")
axes[1].set_ylabel("Trip Count")
axes[1].legend(title="Behavior")
plt.tight_layout()
plt.show()
"""),
    md_cell("""## 4. Sensor Sampling Rate Verification
In UAH-DriveSet:
- IMU (`RAW_ACCELEROMETERS.txt`): ~10 Hz (interval $\Delta t \\approx 0.1$ s)
- GPS (`RAW_GPS.txt`): ~1 Hz (interval $\Delta t \\approx 1.0$ s)
"""),
    code_cell("""sample_trip = trips[0]
imu_df = sample_trip['merged']
dt_imu = np.diff(imu_df['timestamp'].values)
print(f"IMU Mean Sampling Interval: {np.mean(dt_imu):.3f} s (Freq: {1/np.mean(dt_imu):.1f} Hz)")
print(f"IMU Timestamp range: {imu_df['timestamp'].min():.1f}s to {imu_df['timestamp'].max():.1f}s")
""")
])

# ===========================================================================
# 02_preprocessing.ipynb
# ===========================================================================
nb02 = make_notebook([
    md_cell("""# Phase 2: Sensor Preprocessing, Synchronization & Normalization

**Objective**: Synchronize multi-modal IMU (~10 Hz) and GPS (~1 Hz) streams, filter sensor high-frequency noise with a 4th-order Butterworth filter, clip outliers, and establish a leak-free normalization pipeline.
"""),
    code_cell("""import sys
sys.path.insert(0, '..')
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.preprocessing import (
    load_accelerometers,
    load_gps,
    synchronise_sensors,
    butterworth_lowpass,
    clip_outliers_iqr,
    handle_missing_values,
    SensorNormaliser,
)

%matplotlib inline
"""),
    md_cell("""## 1. Sensor Synchronization
GPS (~1 Hz) is linearly interpolated onto IMU (~10 Hz) timestamps, attaching GPS speed and coordinates to every inertial measurement.
"""),
    code_cell("""trip_dir = Path('../data/raw/UAH-DriveSet/D1/20151110175712-16km-D1-NORMAL1-SECONDARY')
raw_imu = load_accelerometers(trip_dir / 'RAW_ACCELEROMETERS.txt')
raw_gps = load_gps(trip_dir / 'RAW_GPS.txt')

print(f"Raw IMU rows: {len(raw_imu)}, Raw GPS rows: {len(raw_gps)}")
synced = synchronise_sensors(raw_imu, raw_gps)
print(f"Synchronized rows: {len(synced)}")
display(synced[['timestamp', 'ax_filt', 'ay_filt', 'speed_kmh', 'latitude', 'longitude']].head())
"""),
    md_cell("""## 2. Low-Pass Butterworth Filtering (2.0 Hz Cutoff)"""),
    code_cell("""t = synced['timestamp'].values[:300]
raw_signal = synced['ax_raw'].values[:300]
filtered_signal = butterworth_lowpass(raw_signal, cutoff_hz=2.0, fs_hz=10.0)

plt.figure(figsize=(11, 4))
plt.plot(t, raw_signal, label="Raw Sensor Signal (ax_raw)", alpha=0.5, color="gray")
plt.plot(t, filtered_signal, label="Butterworth Filtered (2.0 Hz)", color="#2980b9", lw=1.5)
plt.title("Sensor Low-Pass Butterworth Noise Filtering", fontweight="bold")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (g)")
plt.legend()
plt.show()
"""),
    md_cell("""## 3. Data Leakage-Free Normalization
`SensorNormaliser` fits only on training trips and transforms unseen validation and test trips.
"""),
    code_cell("""normaliser = SensorNormaliser(columns=['ax_filt', 'ay_filt', 'az_filt', 'speed_kmh'])
train_sample = synced.iloc[:400].copy()
test_sample = synced.iloc[400:].copy()

train_norm = normaliser.fit_transform(train_sample)
test_norm = normaliser.transform(test_sample)

print(f"Train normalized mean: {train_norm['ax_filt'].mean():.4f}, std: {train_norm['ax_filt'].std():.4f}")
print(f"Test transformed mean: {test_norm['ax_filt'].mean():.4f}, std: {test_norm['ax_filt'].std():.4f}")
""")
])

# ===========================================================================
# 03_feature_extraction.ipynb
# ===========================================================================
nb03 = make_notebook([
    md_cell("""# Phase 3: Time-Window Feature Extraction & Kinematics

**Objective**: Extract kinematic feature vectors across sliding windows (2–5 seconds).
Kinematic features extracted:
- **Longitudinal acceleration** ($a_x$): acceleration and braking
- **Lateral acceleration** ($a_y$): turning forces
- **Vertical acceleration** ($a_z$): road surface irregularities
- **Jerk**: $J(t) = \\frac{a(t) - a(t-\\Delta t)}{\\Delta t}$ (rate of acceleration change)
- **Speed & speed differentials**
- **Acceleration variance**: driving stability
"""),
    code_cell("""import sys
sys.path.insert(0, '..')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.features import (
    WindowConfig,
    compute_jerk,
    extract_all_features,
    FEATURE_COLUMNS,
)
from src.preprocessing import load_all_trips
%matplotlib inline
"""),
    md_cell("""## 1. Mathematical Jerk Formulation"""),
    code_cell("""dt = 0.1
sample_accel = np.array([0.0, 0.02, 0.05, 0.15, 0.35, 0.10, 0.0, -0.20, -0.40, -0.05])
jerk = compute_jerk(sample_accel, dt=dt)

fig, ax1 = plt.subplots(figsize=(9, 3.5))
ax1.plot(sample_accel, 'o-', color='#2980b9', label='Acceleration a(t) [g]')
ax1.set_ylabel('Accel (g)', color='#2980b9')
ax2 = ax1.twinx()
ax2.plot(jerk, 's--', color='#e74c3c', label='Jerk J(t) [g/s]')
ax2.set_ylabel('Jerk (g/s)', color='#e74c3c')
plt.title("Acceleration vs Jerk J(t) = Δa / Δt", fontweight="bold")
plt.show()
"""),
    md_cell("""## 2. Extract Features Across Dataset"""),
    code_cell("""trips = load_all_trips('../data/raw/UAH-DriveSet')
cfg = WindowConfig(window_seconds=3.0, overlap=0.5)
feat_df = extract_all_features(trips, cfg)
print(f"Total extracted windows: {len(feat_df)}")
display(feat_df[FEATURE_COLUMNS].head())
"""),
    md_cell("""## 3. Correlation Heatmap"""),
    code_cell("""plt.figure(figsize=(10, 8))
corr = feat_df[FEATURE_COLUMNS].corr()
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
plt.title("Sensor Feature Correlation Matrix", fontweight="bold")
plt.tight_layout()
plt.show()
""")
])

# ===========================================================================
# 04_sequence_construction.ipynb
# ===========================================================================
nb04 = make_notebook([
    md_cell("""# Phase 4: Sequence Construction & Chronological Partitioning

**Objective**: Convert windowed feature vectors into temporal sequences:
$$X = [X_1, X_2, \\dots, X_T], \\quad X_t \\in \\mathbb{R}^D$$
Sequences preserve strict chronological ordering. No window shuffling is allowed.
"""),
    code_cell("""import sys
sys.path.insert(0, '..')
import pandas as pd
import numpy as np

from src.sequences import (
    build_sequences_from_features,
    SequenceNormaliser,
    split_sequences_chronologically,
)

feat_df = pd.read_csv('../data/processed/features_all.csv')
print(f"Loaded {len(feat_df)} feature rows.")
"""),
    md_cell("""## 1. Sequence Assembly per Trip"""),
    code_cell("""sequences, lengths, meta = build_sequences_from_features(feat_df)
print(f"Constructed {len(sequences)} trip sequences.")
print(f"Sample sequence shape: {sequences[0].shape} (Time windows x Features)")
"""),
    md_cell("""## 2. Chronological Train / Val / Test Partitioning
Split recordings chronologically (70% train, 15% validation, 15% test) with gap buffer between splits.
"""),
    code_cell("""(train_seqs, val_seqs, test_seqs,
 train_meta, val_meta, test_meta) = split_sequences_chronologically(
    sequences, meta, train_frac=0.70, val_frac=0.15, gap=1
)

print(f"Train sequences: {len(train_seqs)} ({sum(len(s) for s in train_seqs)} windows)")
print(f"Val sequences:   {len(val_seqs)} ({sum(len(s) for s in val_seqs)} windows)")
print(f"Test sequences:  {len(test_seqs)} ({sum(len(s) for s in test_seqs)} windows)")
"""),
    md_cell("""## 3. Normalization without Data Leakage"""),
    code_cell("""normaliser = SequenceNormaliser()
train_norm = normaliser.fit_transform(train_seqs)
val_norm = normaliser.transform(val_seqs)
test_norm = normaliser.transform(test_seqs)

print("Normalization complete. Feature scaler fitted strictly on training sequences.")
""")
])

# ===========================================================================
# 05_hmm_training.ipynb
# ===========================================================================
nb05 = make_notebook([
    md_cell("""# Phase 5: Gaussian Hidden Markov Model (HMM) Training

**Objective**: Formulate and train a Gaussian HMM on temporal driving sequences:
- **Hidden states $S = \\{s_1, \\dots, s_K\\}$**: Underlying driving conditions (e.g. stable cruising, harsh acceleration, hard braking, sharp turn)
- **Observations $O_t \\in \\mathbb{R}^D$**: Windowed kinematic feature vectors
- **Initial distribution $\\pi$**: $P(q_1 = s_i)$
- **Transition matrix $A$**: $A_{ij} = P(q_{t+1} = s_j \\mid q_t = s_i)$
- **Emission distributions $B$**: $b_i(O_t) = \\mathcal{N}(O_t; \\mu_i, \\Sigma_i)$
"""),
    code_cell("""import sys
sys.path.insert(0, '..')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.sequences import build_sequences_from_features, SequenceNormaliser, split_sequences_chronologically
from src.hmm_model import train_gaussian_hmm, build_state_label_map, sweep_n_states, save_model

feat_df = pd.read_csv('../data/processed/features_all.csv')
sequences, lengths, meta = build_sequences_from_features(feat_df)
(train_seqs, val_seqs, test_seqs,
 train_meta, val_meta, test_meta) = split_sequences_chronologically(sequences, meta, train_frac=0.7, val_frac=0.15, gap=1)

normaliser = SequenceNormaliser()
train_norm = normaliser.fit_transform(train_seqs)
val_norm = normaliser.transform(val_seqs)
"""),
    md_cell("""## 1. Hyperparameter Sweep: Number of Hidden States $K \\in \\{3, 4, 5\\}$"""),
    code_cell("""val_labels = [m['label'] for m in val_meta]
sweep_df = sweep_n_states(train_norm, val_norm, val_labels, n_states_list=[3, 4, 5])
display(sweep_df)

fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
axes[0].plot(sweep_df['n_states'], sweep_df['accuracy']*100, 'o-', color='#2980b9')
axes[0].set_title("Validation Accuracy (%)", fontweight="bold")
axes[0].set_xlabel("K States")

axes[1].plot(sweep_df['n_states'], sweep_df['aic'], 's-', color='#e74c3c')
axes[1].set_title("AIC (Lower is Better)", fontweight="bold")
axes[1].set_xlabel("K States")

axes[2].plot(sweep_df['n_states'], sweep_df['bic'], '^-', color='#27ae60')
axes[2].set_title("BIC (Lower is Better)", fontweight="bold")
axes[2].set_xlabel("K States")
plt.tight_layout()
plt.show()
"""),
    md_cell("""## 2. Train Chosen Gaussian HMM (K=4)"""),
    code_cell("""model = train_gaussian_hmm(train_norm, n_states=4, covariance_type='diag', n_iter=100, random_state=42)
train_labels = [m['label'] for m in train_meta]
state_map = build_state_label_map(model, train_norm, train_labels)

print("Learned Transition Matrix A:")
print(np.round(model.transmat_, 3))
print("\\nState to Behavior Category Mapping:")
print(state_map)
""")
])

# ===========================================================================
# 06_custom_viterbi.ipynb
# ===========================================================================
nb06 = make_notebook([
    md_cell("""# Phase 6: Custom Viterbi Algorithm vs hmmlearn Reference

**Objective**: Implement the Viterbi decoding algorithm from scratch and benchmark it against `hmmlearn`'s reference implementation.

### Mathematical Formulation (Log-Domain Dynamic Programming):
1. **Initialization** ($t = 0$):
   $$\\delta_0(i) = \\log \\pi_i + \\log b_i(O_0), \\quad \\psi_0(i) = 0$$
2. **Recursion** ($t = 1, \\dots, T-1$):
   $$\\delta_t(j) = \\max_{i} \\left[ \\delta_{t-1}(i) + \\log A_{ij} \\right] + \\log b_j(O_t)$$
   $$\\psi_t(j) = \\arg\\max_{i} \\left[ \\delta_{t-1}(i) + \\log A_{ij} \\right]$$
3. **Termination**:
   $$\\log P^* = \\max_{i} \\delta_{T-1}(i), \\quad q_T^* = \\arg\\max_{i} \\delta_{T-1}(i)$$
4. **Backtracking** ($t = T-2, \\dots, 0$):
   $$q_t^* = \\psi_{t+1}(q_{t+1}^*)$$
"""),
    code_cell("""import sys
sys.path.insert(0, '..')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.custom_viterbi import (
    GaussianHMMParams,
    viterbi_decode,
    viterbi_decode_vectorised,
    compare_with_reference,
)
from src.hmm_model import load_model
from src.sequences import build_sequences_from_features, SequenceNormaliser, split_sequences_chronologically

model, state_map = load_model('../results/model.pkl')
params = GaussianHMMParams.from_hmmlearn(model)
print(f"Loaded HMM with K={params.n_states} states and D={params.n_features} features.")
"""),
    md_cell("""## 1. Load Test Sequence & Run Custom Viterbi"""),
    code_cell("""feat_df = pd.read_csv('../data/processed/features_all.csv')
sequences, lengths, meta = build_sequences_from_features(feat_df)
(tr, va, test_seqs, tr_m, va_m, test_meta) = split_sequences_chronologically(sequences, meta, 0.7, 0.15, 1)

norm = SequenceNormaliser()
norm.fit(tr)
test_norm = norm.transform(test_seqs)

test_obs = test_norm[0]
print(f"Decoding test sequence of length T={len(test_obs)}...")

custom_states, custom_lp = viterbi_decode_vectorised(params, test_obs)
ref_lp, ref_states = model.decode(test_obs)

agreement = np.mean(custom_states == ref_states)
print(f"Sequence Agreement: {agreement:.2%}")
print(f"Custom Log Prob: {custom_lp:.4f} vs Ref Log Prob: {ref_lp:.4f}")
"""),
    md_cell("""## 2. Benchmark Across All Test Sequences"""),
    code_cell("""cmp_df = pd.read_csv('../results/model_comparison/viterbi_comparison.csv')
display(cmp_df)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].bar(range(len(cmp_df)), cmp_df['custom_time_ms'], width=0.35, label='Custom Viterbi', color='#3498db')
axes[0].bar(np.arange(len(cmp_df)) + 0.35, cmp_df['ref_time_ms'], width=0.35, label='hmmlearn decode()', color='#95a5a6')
axes[0].set_ylabel("Inference Time (ms)")
axes[0].set_title("Latency Comparison", fontweight="bold")
axes[0].legend()

axes[1].bar(range(len(cmp_df)), cmp_df['sequence_agreement']*100, color='#2ecc71', edgecolor='black')
axes[1].set_ylabel("Path Agreement (%)")
axes[1].set_ylim(0, 105)
axes[1].set_title("Custom vs Reference Agreement", fontweight="bold")
plt.tight_layout()
plt.show()
""")
])

# ===========================================================================
# 07_model_evaluation.ipynb
# ===========================================================================
nb07 = make_notebook([
    md_cell("""# Phase 7: Model Evaluation & Road Safety Event Detection

**Objective**: Comprehensive evaluation across 3 required metrics:
1. **Accuracy**: Overall driving behavior classification correctness
2. **Macro F1**: Unweighted mean across NORMAL, DROWSY, AGGRESSIVE classes
3. **Event Recall**: Percentage of ground-truth unsafe events detected from `EVENTS_INERTIAL.txt`
"""),
    code_cell("""import sys
sys.path.insert(0, '..')
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.evaluation import evaluate_classification, compute_event_recall
%matplotlib inline
"""),
    md_cell("""## 1. Classification Metrics & Confusion Matrix"""),
    code_cell("""report_df = pd.read_csv('../results/metrics/classification_report.csv', index_index=False if 'index' in open('../results/metrics/classification_report.csv').readline() else None)
print("Classification Report:")
display(pd.read_csv('../results/metrics/classification_report.csv'))

# Display confusion matrix
from PIL import Image
cm_img = Image.open('../results/figures/confusion_matrix.png')
plt.figure(figsize=(7, 6))
plt.imshow(cm_img)
plt.axis('off')
plt.show()
"""),
    md_cell("""## 2. Event Recall Evaluation against EVENTS_INERTIAL.txt"""),
    code_cell("""with open('../results/metrics/event_metrics.json') as f:
    ev_metrics = json.load(f)

print("Event Detection Metrics:")
for k, v in ev_metrics.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.2%}" if v <= 1 else f"  {k}: {v:.2f}")
    else:
        print(f"  {k}: {v}")
"""),
    md_cell("""## 3. Comparison with Naive Threshold Baseline"""),
    code_cell("""with open('../results/metrics/baseline_comparison.json') as f:
    base_cmp = json.load(f)

print("HMM vs Baseline Performance:")
print(f"  HMM Accuracy:      {base_cmp['hmm_accuracy']:.1%}")
print(f"  Baseline Accuracy: {base_cmp['baseline_accuracy']:.1%}")
print(f"  HMM Macro F1:      {base_cmp['hmm_macro_f1']:.1%}")
print(f"  Baseline Macro F1: {base_cmp['baseline_macro_f1']:.1%}")
"""),
    md_cell("""## 4. GPS Route and Event Visualization"""),
    code_cell("""route_img = Image.open('../results/figures/trip_gps_route_events.png')
plt.figure(figsize=(8, 7))
plt.imshow(route_img)
plt.axis('off')
plt.show()
""")
])


notebooks = {
    "01_dataset_exploration.ipynb": nb01,
    "02_preprocessing.ipynb": nb02,
    "03_feature_extraction.ipynb": nb03,
    "04_sequence_construction.ipynb": nb04,
    "05_hmm_training.ipynb": nb05,
    "06_custom_viterbi.ipynb": nb06,
    "07_model_evaluation.ipynb": nb07,
}

for name, nb in notebooks.items():
    path = NOTEBOOKS_DIR / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Generated {path}")

print("All 7 notebooks successfully generated.")
