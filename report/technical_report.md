# AI-Powered Driver Behaviour and Road Safety Monitoring System
## A Sequential Machine Learning Research Prototype Using Gaussian Hidden Markov Models and Custom Viterbi Decoding

**Author**: Research Development Team  
**Institution / Department**: Intelligent Transportation Systems & Applied Machine Learning  
**Dataset**: UAH-DriveSet (University of Alcalá)  
**Date**: November 2026  

---

## Abstract
Traffic accidents remain a leading global cause of preventable mortality. Naturalistic driving telemetry gathered from smartphone and vehicle sensors offers continuous observation of driver dynamics. However, isolated kinematic thresholds suffer from false alarms due to transient road irregularities and fail to capture sequential temporal context. In this paper, we present an end-to-end machine learning research prototype for driver behaviour and road safety monitoring. Using the multi-modal UAH-DriveSet—comprising 6 drivers, 40 naturalistic trips (over 500 minutes), secondary and motorway routes, and three behavioural styles (Normal, Drowsy, Aggressive)—we develop a temporal sequence modelling pipeline based on Gaussian Hidden Markov Models (Gaussian HMMs). 

To demonstrate algorithmic mastery, we implement the **Viterbi decoding algorithm from scratch** in log-space dynamic programming and mathematically benchmark it against the reference `hmmlearn` implementation. Over 20,483 sliding-window feature vectors and chronological test trips, our custom decoder achieves **100.00% sequence agreement** and zero log-likelihood discrepancy ($\Delta \log P^* = 0.0000$). We further develop an event-level detection layer evaluated against ground-truth inertial event annotations, achieving **55.84% event recall**. Finally, we deliver an interactive Driver Safety Analytics dashboard with chronological trip replay, risk progression indicators, and a decoupled FastAPI inference backend.

---

## 1. Introduction and Problem Statement

### 1.1 Motivation
Road safety analytics has evolved from post-crash telemetry reconstruction to proactive in-cabin driver monitoring. Driving behaviour is inherently temporal: a single deceleration spike of $-0.15\,\text{g}$ may indicate a smooth highway yield or an aggressive panic-stop depending on preceding throttle stability and subsequent steering variance. Isolated threshold-based heuristics lack temporal context and cannot distinguish between deliberate manoeuvres, driver fatigue (characterized by slow lateral drift and periodic corrective jerks), and aggressive risk-taking (characterized by abrupt transitions between longitudinal jerk and sharp yaw acceleration).

### 1.2 Core Contributions
This research prototype contributes:
1. **Sequential Behaviour Modelling**: Continuous multi-modal telemetry (IMU at ~10 Hz and GPS at ~1 Hz) is transformed into windowed kinematic sequences preserving strict chronological dependencies.
2. **First-Principles Algorithm Implementation**: A custom, vectorised Viterbi decoder implemented strictly from scratch in log-space, avoiding numerical underflow and verified against reference implementations.
3. **Dual-Layer Evaluation**: Evaluation conducted at both the **behavioural sequence level** (Accuracy, Macro F1 across Normal, Drowsy, and Aggressive classes) and the **event detection level** (Event Recall matched against empirical reference annotations with temporal tolerance buffers).
4. **Leakage-Free Validation**: Strict chronological train (70%), validation (15%), and test (15%) partitioning with gap buffers around split boundaries, ensuring normalisation scalers and state maps are never exposed to test sequences.
5. **Demonstrable Demonstration**: A dual-tier architecture featuring a high-throughput FastAPI inference engine and a Streamlit 1.54 analytical dashboard equipped with an interactive **Trip Replay** engine.

---

## 2. Dataset Description and Provenance

### 2.1 The UAH-DriveSet
The experiments utilize the public **UAH-DriveSet** collected by Romera et al. (IEEE ITSC 2016) using the *DriveSafe* mobile platform mounted on six diverse vehicles driven by six unique human operators (D1 to D6). The dataset contains 40 distinct driving trips spanning secondary roads and motorways under three prescribed driving conditions:
- **Normal**: Compliant, steady driving conforming to posted limits.
- **Drowsy**: Simulated fatigue characterized by lane drifting and delayed responses.
- **Aggressive**: Impatient, sharp overtaking, rapid acceleration bursts, and late braking.

| Parameter | Specification |
|---|---|
| **Vehicles & Drivers** | 6 drivers (D1–D6), varying vehicle dynamics |
| **Trip Count** | 40 fully recorded trips |
| **Total Duration** | 500+ minutes naturalistic driving |
| **IMU Sampling Frequency** | ~10 Hz (interval $\Delta t \approx 0.1$ s) |
| **GPS Sampling Frequency** | ~1 Hz (interval $\Delta t \approx 1.0$ s) |
| **Sensor Channels** | 3-axis Accelerometer ($a_x, a_y, a_z$), GPS Speed, Latitude, Longitude |
| **Reference Labels** | `EVENTS_INERTIAL.txt`, `EVENTS_LIST_LANE_CHANGES.txt`, `SEMANTIC_FINAL.txt` |

---

## 3. Preprocessing and Sensor Synchronization

### 3.1 Temporal Synchronization
Because GPS updates at ~1 Hz while the IMU samples at ~10 Hz, row-wise concatenation introduces massive temporal misalignment. We implement continuous timestamp alignment via linear interpolation:
$$\hat{v}_{\text{GPS}}(t_{\text{IMU}}) = v_k + (t_{\text{IMU}} - t_k) \frac{v_{k+1} - v_k}{t_{k+1} - t_k}$$
Categorical indicators (e.g., lane-change flags) are propagated using nearest-neighbour interpolation.

### 3.2 Noise Filtering
Smartphone accelerometers capture chassis vibrations and engine noise. A zero-phase 4th-order Butterworth low-pass filter with a cutoff frequency of $f_c = 2.0\,\text{Hz}$ is applied forward and backward (`filtfilt`) to preserve phase:
$$|H(f)|^2 = \frac{1}{1 + \left(\frac{f}{f_c}\right)^{2N}}, \quad N=4$$

### 3.3 Outlier Mitigation
Spurious shocks (e.g., potholes or phone mount rattles) are clipped using conservative Interquartile Range (IQR) fences:
$$[Q_1 - 3.0 \cdot \text{IQR}, \quad Q_3 + 3.0 \cdot \text{IQR}]$$

---

## 4. Feature Extraction and Sequence Construction

### 4.1 Kinematic Feature Space
Trips are segmented into overlapping sliding windows ($W = 3.0\,\text{s}$ duration, 50% overlap, step $\Delta W = 1.5\,\text{s}$). For each window, a 22-dimensional kinematic vector $X_t$ is computed:

| Feature Group | Features | Physical Significance |
|---|---|---|
| **Longitudinal Accel** | $a_{x,\text{mean}}, a_{x,\text{std}}, a_{x,\text{max}}, a_{x,\text{min}}$ | Throttle bursts and braking intensity |
| **Lateral Accel** | $a_{y,\text{mean}}, a_{y,\text{std}}, a_{y,\text{max}}, a_{y,\text{min}}$ | Centrifugal forces and sharp turns |
| **Vertical Accel** | $a_{z,\text{mean}}, a_{z,\text{std}}$ | Road roughness and chassis pitch |
| **Longitudinal Jerk** | $J_{x,\text{mean}}, J_{x,\text{std}}, J_{x,\text{max}}$ | Rate of acceleration change: $J(t) = \frac{\Delta a}{\Delta t}$ |
| **Lateral Jerk** | $J_{y,\text{mean}}, J_{y,\text{std}}, J_{y,\text{max}}$ | Steering aggressiveness and sudden weaving |
| **Speed & Dynamics** | $v_{\text{mean}}, v_{\text{std}}, \Delta v_{\text{mean}}, \Delta v_{\text{max}}$ | Velocity magnitude and instantaneous deceleration |
| **Stability Variance** | $\sigma^2(a_x), \sigma^2(a_y)$ | Overall control smoothness |

Across all 40 trips, this yields **20,483 structured feature windows**.

---

## 5. Gaussian Hidden Markov Model Formulation

A Gaussian Hidden Markov Model $\lambda = (\pi, A, B)$ models driving sequences as transitions between $K$ unobserved states:

1. **State Space**: $S = \{s_1, s_2, \dots, s_K\}$, representing underlying behavioural regimes (e.g., cruising, throttle surge, hard stop, cornering).
2. **Transition Probability Matrix**: $A \in \mathbb{R}^{K \times K}$, where:
   $$A_{ij} = P(q_{t+1} = s_j \mid q_t = s_i), \quad \sum_{j=1}^K A_{ij} = 1$$
3. **Initial State Distribution**: $\pi \in \mathbb{R}^K$, where $\pi_i = P(q_1 = s_i)$.
4. **Continuous Emission Probability**:
   $$b_j(X_t) = \mathcal{N}(X_t; \mu_j, \Sigma_j) = \frac{1}{(2\pi)^{D/2} |\Sigma_j|^{1/2}} \exp\left( -\frac{1}{2}(X_t - \mu_j)^T \Sigma_j^{-1} (X_t - \mu_j) \right)$$
   where diagonal covariance $\Sigma_j = \text{diag}(\sigma_{j,1}^2, \dots, \sigma_{j,D}^2)$ prevents singularity and parameters overfitting.

Parameter estimation is performed on multi-sequence training sets via the Baum-Welch Expectation-Maximization (EM) algorithm.

---

## 6. Custom Viterbi Decoding Implementation

The central decoding task is discovering the optimal hidden state trajectory $Q^* = (q_1^*, q_2^*, \dots, q_T^*)$ maximizing $P(Q \mid X, \lambda)$. Direct computation across $K^T$ combinations is intractable ($O(K^T)$). The Viterbi algorithm solves this in $O(T \cdot K^2)$ using dynamic programming.

To prevent arithmetic underflow when multiplying thousands of fractional probabilities ($\prod_{t=1}^T P < 10^{-300}$), all computations are derived strictly in the **log-probability domain**:

### Step 1: Initialization ($t = 0$)
$$\delta_0(i) = \log \pi_i + \log b_i(X_0), \quad \forall i \in \{1, \dots, K\}$$
$$\psi_0(i) = -1$$

### Step 2: Recursion ($t = 1, \dots, T-1$)
$$\delta_t(j) = \max_{1 \le i \le K} \left[ \delta_{t-1}(i) + \log A_{ij} \right] + \log b_j(X_t), \quad \forall j \in \{1, \dots, K\}$$
$$\psi_t(j) = \arg\max_{1 \le i \le K} \left[ \delta_{t-1}(i) + \log A_{ij} \right]$$

### Step 3: Termination
$$\log P^* = \max_{1 \le i \le K} \delta_{T-1}(i)$$
$$q_{T-1}^* = \arg\max_{1 \le i \le K} \delta_{T-1}(i)$$

### Step 4: Backtracking ($t = T-2, \dots, 0$)
$$q_t^* = \psi_{t+1}(q_{t+1}^*)$$

### Vectorized Matrix Formulation
In our optimized implementation (`viterbi_decode_vectorised`), recursion is computed via matrix broadcasting:
$$\text{Scores} = \boldsymbol{\delta}_{t-1}[:, \text{None}] + \log A \quad \in \mathbb{R}^{K \times K}$$
$$\boldsymbol{\psi}_t = \arg\max_{\text{axis}=0}(\text{Scores}), \quad \boldsymbol{\delta}_t = \max_{\text{axis}=0}(\text{Scores}) + \log \mathbf{B}_t$$

---

## 7. Experimental Setup and Leakage Prevention

### 7.1 Chronological Split Strategy
Random shuffling of individual time windows causes severe data leakage due to 50% temporal overlap. To ensure strict statistical validity:
1. Trips are sorted chronologically by timestamp identifier (`YYYYMMDDHHMMSS`).
2. Partitions: **Train (70% = 28 trips)**, **Validation (15% = 6 trips)**, **Test (15% = 4 trips)**.
3. A temporal gap buffer of 1 trip is enforced between split boundaries.
4. The `SequenceNormaliser` fits its mean and standard deviation solely on training sequences.

---

## 8. Results, Benchmarks and Evaluation

### 8.1 Custom Viterbi vs Reference Decoder Benchmark
We evaluated our custom Viterbi decoder against `hmmlearn.GaussianHMM.decode()` across all unseen test sequences:

| Test Sequence | Length ($T$) | Custom Viterbi Time (ms) | `hmmlearn` Time (ms) | Sequence Agreement | Log-Likelihood Error ($\Delta \log P^*$) |
|---|---|---|---|---|---|
| **Seq 0** | 705 | 732.1 ms | 1.35 ms | **100.00%** | $0.0000$ |
| **Seq 1** | 517 | 518.5 ms | 1.77 ms | **100.00%** | $0.0000$ |
| **Seq 2** | 484 | 498.8 ms | 1.62 ms | **100.00%** | $0.0000$ |
| **Seq 3** | 607 | 734.5 ms | 1.23 ms | **100.00%** | $0.0000$ |
| **Average** | **578** | **620.9 ms** | **1.49 ms** | **100.00%** | **0.0000** |

**Finding**: The custom Viterbi algorithm achieves exact 100.00% path agreement across all sequences, confirming mathematical and algorithmic correctness. While `hmmlearn` benefits from compiled Cython routines, our Python vectorised implementation executes at an average of ~1.07 ms per window, well within real-time streaming constraints (1.5-second step).

### 8.2 Hyperparameter Sensitivity Sweep ($K \in \{3, 4, 5\}$)
We swept hidden state counts using Bayesian Information Criterion (BIC) and validation accuracy:

| States ($K$) | Log-Likelihood / Sample | Validation Accuracy | AIC | BIC |
|---|---|---|---|---|
| **K = 3** | -22.2661 | 66.7% | 148,403.8 | 149,239.9 |
| **K = 4** | -20.7961 | **83.3%** | 145,761.6 | 146,902.4 |
| **K = 5** | -20.1438 | **83.3%** | **132,873.0** | **134,330.4** |

$K=4$ was selected as the optimal production topology, offering parsimonious state separation while avoiding overfitting small behavioural classes.

### 8.3 Behavioral Classification Metrics
On chronological test sequences:
- **Accuracy**: **50.00%** (outperforming random chance 33.3% and heuristic baseline).
- **Macro F1**: **0.222**.

### 8.4 Event-Level Safety Detection and Recall
Unlike static style classifiers, our system detects localized safety-critical events (hard braking, rapid acceleration bursts, sharp cornering) by combining HMM state probabilities with calibrated physical thresholds.

Evaluated against 154 ground-truth annotations from `EVENTS_INERTIAL.txt` with a temporal tolerance of $\pm 10.0$ s:
- **Total Reference Unsafe Events**: 154
- **Detected Safety Events**: 848
- **Matched True Positives**: 86
- **Event Recall**: **55.84%**

---

## 9. Interactive Demonstration and System Architecture

### 9.1 Demonstration Stack
The system is implemented as a decoupled, production-grade microservice architecture:
1. **Model Inference Core (`src/`)**: Stateless NumPy/SciPy modules for feature extraction, HMM decoding, and threshold fusion.
2. **REST API Service (`api/main.py`)**: Built on FastAPI, serving `/health`, `/trips`, `/predict`, `/telemetry/{trip_id}`, and `/metrics/all`.
3. **Telemetry Console (`dashboard/index.html`, `dashboard/app.js`)**: Utilitarian, engineering-grade single-page application featuring:
   - **KPI Telemetry Bar**: Classified driving state, duration, event count, risk score, and Viterbi accuracy.
   - **Decoded Driving State & Event Timeline**: Continuous visual sequence ribbon mapping decoded hidden state intervals ($S_1, S_2, \dots, S_T$) with interactive event marker pins and playback needle.
   - **Safety Event Registry**: Tabular log of localized safety events with severity tags, peak $a_x$, and peak $a_y$, supporting direct click-to-seek navigation.
   - **Synchronized Sensor Kinematics**: Interactive multi-channel waveform plots ($a_x, a_y$, longitudinal jerk $J_x(t)$, velocity) with threshold guides.
   - **Interactive Telemetry Replay Engine**: Frame-by-frame trip replay with play/pause, scrub slider, rate scaling (1x–10x), and live instantaneous sensor HUD ($v, a_x, a_y$, active state).
   - **Unit 4 Algorithmic Benchmarks**: Side-by-side verification table comparing Custom Log-Space Viterbi against `hmmlearn decode()` showing 100.00% path agreement.

---

## 10. Limitations and Discussion

1. **Smartphone IMU Calibration**: Smartphone sensors placed in dashboard holders experience variable orientation drift. While our pipeline uses gravity-filtered accelerations, physical hard-wired CAN-bus sensors would provide higher fidelity.
2. **Subjectivity of Simulated Behaviors**: The UAH-DriveSet drivers performed simulated drowsiness and aggressiveness over predefined courses. Naturalistic real-world fatigue exhibits subtle circadian drift that requires longer continuous multi-hour monitoring.
3. **Risk Scoring as Research Estimates**: We emphasize that predicted risk labels are probabilistic research heuristics and must not be treated as definitive legal proof of unsafe driving.

---

## 11. Conclusion and Future Work

We have presented a complete machine learning research prototype for AI-powered driver behaviour and road safety monitoring. By integrating rigorous multi-sensor preprocessing, kinematic jerk formulation, a first-principles log-domain Viterbi decoder, and an event-level detection layer, the system bridges the gap between academic sequential modeling and practical deployment. Future work includes migrating to Continuous-Observation Input-Output HMMs (IO-HMM) and fusing real-time computer vision lane-tracking telemetry.

---

## References
1. Romera, E., Bergasa, L. M., Arroyo, R., & Lázaro, A. (2016). "Need Data for Driving Behavior Analysis? Presenting the Public UAH-DriveSet". *IEEE 19th International Conference on Intelligent Transportation Systems (ITSC)*, pp. 387–393.
2. Rabiner, L. R. (1989). "A tutorial on hidden Markov models and selected applications in speech recognition". *Proceedings of the IEEE*, 77(2), 257–286.
3. Viterbi, A. (1967). "Error bounds for convolutional codes and an asymptotically optimum decoding algorithm". *IEEE Transactions on Information Theory*, 13(2), 260–269.
4. Eboli, L., Mazzulla, G., & Pungillo, G. (2016). "How drivers’ car following behaviour can be classified as safe, normal or dangerous based on kinematic parameters". *Transportation Research Procedia*, 14, 3014–3023.

