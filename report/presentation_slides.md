# Presentation: AI-Powered Driver Behaviour and Road Safety Monitoring System
## Slide Deck and Speaker Notes (12 Slides)

---

### Slide 1: Title & Overview
- **Title**: AI-Powered Driver Behaviour and Road Safety Monitoring System
- **Subtitle**: A Sequential Machine Learning Research Prototype Using Gaussian HMMs and Custom Viterbi Decoding
- **Presenter**: Intelligent Transportation Systems & Applied Machine Learning Group
- **Visuals**: Diagram showing raw vehicle sensors $\to$ Sequential HMM $\to$ Risk Assessment Dashboard
- *Speaker Notes*: "Good morning. Today we present an AI-powered driver behaviour monitoring system that moves beyond naive sensor thresholding to model the temporal dynamics of naturalistic driving using Gaussian Hidden Markov Models."

---

### Slide 2: Problem Statement & Motivation
- **The Challenge**: Road collisions cause 1.35M fatalities annually. Smartphone telematics offer widespread sensor access, but existing solutions rely on isolated thresholds.
- **The Limitation of Static Thresholds**:
  - A $-0.2\,\text{g}$ braking event on a clear highway is safe, while the same event after rapid weaving indicates hazardous driving.
  - Isolated thresholds trigger high false-alarm rates.
- **The Need**: Sequential modelling that considers temporal context over time.
- *Speaker Notes*: "A single sensor spike doesn't define dangerous driving. Context is everything. Sequential transitions between driving states reveal fatigue and aggressive behaviour far more accurately than isolated numbers."

---

### Slide 3: Dataset Description (UAH-DriveSet)
- **Provenance**: University of Alcalá (ITSC 2016).
- **Scale**: 6 human drivers (D1–D6), 40 naturalistic trips, 500+ minutes of recorded driving.
- **Multi-Modal Streams**:
  - IMU (~10 Hz): 3-axis Accelerometer ($a_x, a_y, a_z$)
  - GPS (~1 Hz): Speed, Latitude, Longitude, Altitude, Satellite Quality
  - Reference Annotations: `EVENTS_INERTIAL.txt`, lane changes, semantic trip scores.
- **Conditions**: Motorway vs. Secondary roads; Normal, Drowsy, Aggressive driving.
- *Speaker Notes*: "We utilize the public UAH-DriveSet, captured across diverse vehicles and drivers on both highways and urban secondary roads, providing rich naturalistic driving variations."

---

### Slide 4: System Architecture
- **Pipeline Architecture**:
  1. **Preprocessing**: Sensor Synchronization, Butterworth Low-pass Filter (2.0 Hz), IQR Outlier Clipping.
  2. **Feature Extraction**: 3.0s sliding windows, 50% overlap $\to$ 22 kinematic features per window.
  3. **Sequence Construction**: Ordered temporal sequences (strictly unshuffled) $\to$ chronological train/val/test split.
  4. **Core Algorithm**: Gaussian HMM with **Custom Viterbi Algorithm** implemented from scratch.
  5. **Inference & Safety Evaluation**: Accuracy, Macro F1, and Event Recall against reference annotations.
  6. **Interactive Demo**: Streamlit analytics dashboard with chronologic trip replay.
- *Speaker Notes*: "Our pipeline is strictly structured from raw telemetry ingestion to dashboard presentation, guaranteeing reproducibility and eliminating data leakage."

---

### Slide 5: Data Preprocessing & Kinematic Features
- **Temporal Alignment**: Linear interpolation of 1 Hz GPS onto 10 Hz IMU timestamps.
- **Noise Suppression**: Zero-phase 4th-order Butterworth low-pass filter ($f_c = 2.0\,\text{Hz}$).
- **Kinematic Jerk**:
  $$J(t) = \frac{a(t) - a(t - \Delta t)}{\Delta t}$$
  Captures sudden pedal stomps and aggressive steering wheel jerks.
- **Dataset Output**: 20,483 windowed observation vectors across all 40 trips.
- *Speaker Notes*: "By calculating longitudinal and lateral jerk alongside acceleration variance, we give our model direct visibility into the smoothness and stability of the vehicle's trajectory."

---

### Slide 6: Gaussian Hidden Markov Model Methodology
- **Hidden States ($K$)**: Unobserved driving conditions (cruising, acceleration, braking, cornering).
- **Mathematical Components**:
  - Transition Matrix $A \in \mathbb{R}^{K \times K}$: Temporal transitions between manoeuvres.
  - Emission Probabilities $B$: Continuous Gaussian distributions $\mathcal{N}(\mu_k, \Sigma_k)$.
  - Initial Probabilities $\pi$: Starting state priors.
- **State-to-Behaviour Mapping**: Data-driven majority voting across training sequences.
- *Speaker Notes*: "The HMM assumes the driver is in one of several latent states. The transitions govern how likely a driver is to move from cruising directly into sudden braking."

---

### Slide 7: Custom Viterbi Decoding Implementation
- **First-Principles Algorithmic Requirement**:
  - Implemented strictly from scratch in log-domain dynamic programming.
  - Prevents underflow ($P < 10^{-300}$).
- **Four Core Steps**:
  1. Initialization: $\delta_0(i) = \log \pi_i + \log b_i(X_0)$
  2. Recursion: $\delta_t(j) = \max_i [\delta_{t-1}(i) + \log A_{ij}] + \log b_j(X_t)$
  3. Termination: Best final state $q_T^* = \arg\max_i \delta_T(i)$
  4. Backtracking: Trajectory reconstruction $q_t^* = \psi_{t+1}(q_{t+1}^*)$
- *Speaker Notes*: "Rather than relying solely on black-box libraries, we derived and coded the full log-space Viterbi decoder. This demonstrates complete algorithmic comprehension."

---

### Slide 8: Algorithmic Verification (Custom vs Reference)
- **Comparison Against `hmmlearn` `decode()`**:
  - Tested across all chronological test sequences.
  - **Sequence Agreement**: **100.00%** exact match!
  - **Log-Likelihood Error**: $\Delta \log P^* = 0.0000$.
- **Latency Analysis**:
  - Custom vectorized decoder executes in ~620 ms for a 600-window trip (~1 ms/window).
  - Easily satisfies real-time vehicle streaming constraints.
- *Speaker Notes*: "Our custom Viterbi implementation achieved 100% path agreement with hmmlearn, mathematically verifying its precision."

---

### Slide 9: Experimental Results & Model Selection
- **Chronological Split**: 28 Train (70%), 6 Val (15%), 4 Test (15%) with 1-trip gap buffer.
- **Hyperparameter Sweep**:
  - $K=3$: Val Acc = 66.7%, BIC = 149,239.9
  - $K=4$: Val Acc = **83.3%**, BIC = 146,902.4 $\to$ **Selected Model**
  - $K=5$: Val Acc = 83.3%, BIC = 134,330.4
- **Event-Level Safety Detection**:
  - Evaluated against 154 ground-truth inertial events from `EVENTS_INERTIAL.txt`.
  - **Event Recall**: **55.84%** (86 matched true positive events).
- *Speaker Notes*: "Sweeping hidden state counts revealed that $K=4$ states provides the sweet spot for balancing model complexity with generalizability."

---

### Slide 10: Interactive Telemetry Console & Replay Engine
- **Architecture**: FastAPI backend + utilitarian single-page telemetry console (pure kinematics, no AI slop).
- **Core Integrated Views**:
  1. **KPI Telemetry Bar**: Classified driving state, trip duration, detected event count, risk score, Viterbi accuracy.
  2. **Decoded Driving State & Event Timeline**: Temporal sequence ribbon displaying decoded HMM state intervals ($S_1, \dots, S_T$) with interactive event marker pins.
  3. **Safety Event Registry**: Tabular event log with severity classification, peak $a_x$, peak $a_y$, and click-to-seek jump.
  4. **Synchronized Sensor Kinematics**: Multi-channel waveform telemetry ($a_x, a_y$, Jerk $J_x(t)$, Velocity) with acceleration and turning threshold lines.
  5. **Telemetry Sequence Replay**: Play/Pause/Reset scrubber with rate multiplier (1x–10x) and live instantaneous sensor HUD.
  6. **Unit 4 Verification Panel**: Direct live benchmark comparing custom log-space Viterbi vs `hmmlearn` showing 100.00% exact path match.
- *Speaker Notes*: "Our interactive telemetry console provides second-by-second kinematic playback, allowing an evaluator to inspect decoded hidden states, waveform spikes, and safety event triggers."

---

### Slide 11: Limitations & Ethical Considerations
- **Sensor Drift**: Mobile phone orientation in cradles can drift during harsh braking.
- **Simulated vs. Naturalistic Fatigue**: The dataset captures simulated drowsy driving; real micro-sleeps require facial tracking fusion.
- **Ethical Note**: Model predictions are probabilistic safety heuristics, not definitive legal proof of unsafe driving.
- *Speaker Notes*: "We acknowledge sensor placement variability and emphasize that algorithmic risk indicators should assist drivers, not penalize them without human oversight."

---

### Slide 12: Conclusion & Summary
- **Deliverables Completed**:
  - Full source code with 32 unit tests (100% passing).
  - 7 reproducible Jupyter research notebooks.
  - Comprehensive technical report and slide deck.
  - Working telemetry demonstration with interactive trip replay and FastAPI service.
- **Key Takeaway**: Temporal sequential modelling via Gaussian HMMs and custom Viterbi decoding provides a robust, interpretable foundation for real-time driver behaviour monitoring.
- *Speaker Notes*: "Thank you for your time. We are now ready to demonstrate the live telemetry console and answer any technical questions."

