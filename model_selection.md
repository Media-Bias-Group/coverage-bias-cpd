# Model Selection Methodology for Structural Change Point Detection

## Task Definition

The objective of this project is to detect **structural change points** in news coverage bias patterns, i.e. identifying points in time where relevant statistical or latent properties of the underlying time series change.

Following the definition by Aminikhanghahi and Cook (2017):

> “Change point detection (CPD) is the problem of finding abrupt changes in data when a property of the time series changes. Segmentation, edge detection, event detection and anomaly detection are similar concepts which are occasionally applied as well as change point detection.”

Our task therefore falls specifically into the domain of **Change Point Detection (CPD)**. While anomaly detection methods can in principle be adapted for CPD tasks, they are not always directly optimized for detecting structural shifts in latent representations over time.

---

# Conceptual Framing of the Task

We further narrowed down the conceptual framing of the task in order to systematically select appropriate models.

The target task can be characterized as:

- **Offline** change point detection
- **Multivariate** time series analysis
- **Unsupervised** learning
- Detection of **structural changes in latent representations**

More specifically, our goal is 

> detecting shifts in the latent structure of the embedding space.

---

# Approaches to Change Point Detection

The surveyed literature generally distinguishes between three major methodological approaches:

1. **Forecasting-based approaches**
2. **Reconstruction-based approaches**
3. **Representation-based approaches**

## Forecasting and Reconstruction Approaches

Forecasting and reconstruction models detect change points indirectly through:

- prediction errors,
- reconstruction loss,
- or deviations from expected temporal patterns.

This means that identified change points do not necessarily correspond to actual structural changes in the latent embedding space.

These approaches are particularly sensitive to:

- noisy input data,
- non-stationary temporal dynamics,
- and unstable forecasting targets.

Since the GDELT dataset is expected to contain substantial noise and volatility, forecasting and reconstruction methods may struggle to robustly identify meaningful structural shifts.

## Representation-based Approaches

Representation-based methods instead aim to detect changes directly in the learned latent representation space.

This aligns closely with the actual research objective:

- detecting structural shifts in coverage patterns,
- identifying changes in latent semantic structure,
- and modeling evolving embedding distributions over time.

Therefore, representation-based approaches are conceptually the best fit for the task.

---

# Systematic Model Selection

The model selection process follows a systematic filtering strategy based on the requirements derived from the task definition.

Using existing survey papers on Change Point Detection and Time Series Anomaly Detection, all reviewed models were evaluated according to the following criteria:

- Offline setting
- Multivariate capability
- Unsupervised learning
- Representation-based methodology

Only models satisfying all criteria were retained for further consideration.

---

# Relation Between Anomaly Detection and CPD

Anomaly detection models can sometimes be transferred to CPD tasks.

However, their applicability depends strongly on the specific model architecture.

Some anomaly detection models produce an anomaly score for a temporal window. By defining a threshold, anomalies exceeding this threshold can potentially be interpreted as change points.

Nevertheless, this transfer is not always straightforward because:

- threshold selection is non-trivial,
- anomaly scores may not correspond to structural changes,
- and temporal localization can become ambiguous.

Therefore, anomaly detection models should currently be treated as secondary candidates rather than the primary methodological focus.

---

# Methodological Justification

The methodological motivation for the final model selection can be summarized:

1. The task specifically requires detecting structural changes in latent representations.
2. GDELT data is likely noisy and difficult to model through forecasting or reconstruction objectives.
3. Representation-based approaches directly optimize for changes in embedding structure.
4. Models were systematically selected from survey literature according to:
   - offline capability,
   - multivariate processing,
   - unsupervised learning,
   - and representation-based modeling.

