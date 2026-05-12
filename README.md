# Model Poisoning-Resilient Split Federated Learning (SFL)

## 📌 Overview

This project investigates Federated Learning (FL), Split Learning (SL), and Split Federated Learning (SFL), with a focus on robustness against poisoning attacks under highly heterogeneous Non-IID environments.

The work progresses from baseline FL implementations to controlled FL vs SFL equivalence studies, followed by adversarial attack simulations and multiple defence mechanisms. The experiments explore how SFL behaves compared to standard FL under extreme label skew, deeper architectures, and poisoning attacks.

---

## 🎯 Objectives

* Implement baseline Federated Learning
* Evaluate IID and Non-IID training behaviour
* Implement Split Learning and Split Federated Learning (SFL)
* Validate FL vs SFL mathematical equivalence
* Study the effect of deeper architectures under Non-IID settings
* Simulate poisoning attacks (SGA)
* Develop and evaluate defence mechanisms:

  * Median/MAD filtering
  * Cosine similarity filtering
  * Hybrid defence mechanisms
  * SFL-aware server-side defences

---

## 📁 Project Structure

```text
notebooks/
│ 01_fl_baseline_mnist.ipynb
│ 02_fl_multidataset_mnist_fmnist_cifar10.ipynb
│ 03_split_learning_vs_full_model.ipynb
│ 04_fl_vs_sl_equivalence_study_3clients_MNIST.ipynb
│ 05A_fl_vs_sfl_noniid_one_label_2LAYER.ipynb
│ 05B_fl_vs_sfl_noniid_one_label_3LAYER.ipynb
│ 05C_fl_vs_sfl_noniid_one_label_FMNIST.ipynb
│ 05D_fl_vs_sfl_noniid_one_label_CIFAR10.ipynb
│ 05E_fl_vs_sfl_noniid_one_label_4LAYER.ipynb
│ 05F_fl_vs_sfl_sga_attack.ipynb
│ 05G_fl_sfl_median_mad_defence.ipynb
│ 05H_fl_sfl_cosine_defence.ipynb
│ 05I_combined_defence.ipynb
│ 05J_sfl_aware_layer_specific_defence.ipynb

results/
│ plots/
│ csv/

weights/
│ fl/
│ sfl/
```

---

# 🧠 Notebook Summaries

## 01 – FL Baseline (MNIST)

* Standard Federated Learning using MNIST
* FedAvg aggregation
* Tracks loss and accuracy

**Purpose:** Establish baseline FL behaviour

---

## 02 – Multi-Dataset FL Analysis

* FL experiments on:

  * MNIST
  * Fashion-MNIST
  * CIFAR-10
* IID and Non-IID configurations:

  * 1, 2, 4, and 6 labels per client

**Key Insight:**
Performance degrades significantly as:

* dataset complexity increases,
* and client label diversity decreases.

---

## 03 – Split Learning vs Full Model

* Implements Split Learning with:

  * 2 client layers
  * 3 server layers
* Compares against a centralized full model

**Key Insight:**
Split Learning can reproduce centralized learning behaviour under IID conditions.

---

## 04 – FL vs SFL Equivalence Study

* Controlled comparison between FL and SFL
* Uses:

  * identical initialization,
  * synchronized batching,
  * separate client/server aggregation
* Tracks:

  * accuracy,
  * loss,
  * L2 distance

**Key Insight:**
FL and SFL can remain mathematically equivalent when training conditions are synchronized.

---

## 05A – SFL under Extreme Non-IID

* One label per client
* 2-layer architecture (closest to TwoNN baseline)

**Key Insight:**
Simple shallow architectures perform best under extreme Non-IID settings.

---

## 05B – Deeper 3-Layer SFL

* 1 client layer + 2 server layers

**Key Insight:**
Increasing model depth reduces convergence quality and final accuracy under one-label-per-client distributions.

---

## 05C – FMNIST under Extreme Non-IID

* Same 3-layer split architecture applied to Fashion-MNIST

**Key Insight:**
Training instability increases substantially with dataset complexity.

---

## 05D – CIFAR-10 under Extreme Non-IID

* Same 3-layer split architecture applied to CIFAR-10

**Key Insight:**
CIFAR-10 becomes highly difficult under extreme label skew and shallow MLP architectures.

---

## 05E – Deeper 4-Layer Architecture

* 2 client layers + 2 server layers

**Key Insight:**
Deeper architectures further worsen convergence under extreme Non-IID settings.

---

## 05F – Poisoning Attack (SGA)

* Implements Sign Gradient Attack (SGA)

**Key Insight:**
Standard FL collapses under attack, while SFL shows partial resilience and continues learning.

---

## 05G – Median/MAD Defence

* Introduces Median + MAD anomaly filtering

**Key Insight:**
Median/MAD defence significantly improves robustness and prevents catastrophic collapse.

---

## 05H – Cosine Similarity Defence

* Historical cosine similarity filtering
* Warm-up rounds before defence activation

**Key Insight:**
Cosine similarity alone is unstable and insufficient as a standalone defence.

---

## 05I – Combined Defence

* Hybrid:

  * Median/MAD
  * Historical cosine similarity

**Key Insight:**
Combined defences improve robustness and stability compared to cosine-only filtering.

---

## 05J – SFL-Aware Layer-Specific Defence

* Uses:

  * server-observable gradients,
  * server-side loss behaviour,
  * layer-specific anomaly detection

**Key Insight:**
SFL may provide additional defensive capabilities not naturally available in standard FL.

---

# 📊 Results Summary

## 🔹 FL vs SFL Equivalence

* FL and SFL remained numerically equivalent when:

  * initialization,
  * data ordering,
  * aggregation,
  * and optimization settings were synchronized.
* L2 distance was maintained near zero.

---

## 🔹 Effect of Non-IID Data

* Extreme label skew severely degrades convergence.
* One-label-per-client configurations are particularly difficult.
* Dataset complexity strongly impacts stability:

  * MNIST performed best,
  * FMNIST showed moderate instability,
  * CIFAR-10 struggled significantly.

---

## 🔹 Effect of Model Depth

* Shallow models performed better under extreme Non-IID conditions.
* Increasing depth reduced:

  * convergence speed,
  * stability,
  * and final accuracy.

---

## 🔹 Poisoning Attack Behaviour

* Standard FL was highly vulnerable to SGA attacks.
* FL frequently collapsed completely.
* SFL consistently showed improved resilience and partial recovery.

---

## 🔹 Defence Mechanism Findings

### Median/MAD Defence

* Most stable standalone defence
* Best balance between:

  * attack filtering,
  * convergence stability,
  * and learning performance

### Cosine Similarity Defence

* Highly sensitive to noisy gradients
* Required warm-up and historical references
* Insufficient as a standalone defence

### Combined Defence

* Improved robustness compared to cosine-only filtering
* Best performance when:

  * Median/MAD acted as primary defence,
  * cosine similarity acted as secondary validation

### SFL-Aware Defence

* Leveraged server-observable information
* Showed strongest overall robustness and stability

---

# ⚠️ Key Observations

* Non-IID data fundamentally changes distributed learning dynamics.
* Deeper architectures amplify instability under label skew.
* Standard FL aggregation alone is insufficient against strong poisoning attacks.
* SFL introduces additional observability through server-side gradients and activations.
* These additional signals may enable stronger defence mechanisms.

---

# 🚀 Future Work

* Extend experiments to:

  * multiple attackers,
  * adaptive attacks,
  * targeted poisoning,
  * activation manipulation attacks
* Explore:

  * transformer-based architectures,
  * CNN-based SFL,
  * adaptive anomaly thresholds,
  * temporal defence models,
  * trust-based aggregation systems
* Evaluate scalability under larger client populations and realistic federated environments.

---

# 🧠 Conclusion

This project demonstrates that Split Federated Learning can reproduce Federated Learning behaviour under controlled conditions while also exhibiting improved robustness characteristics under poisoning attacks.

The experiments suggest that SFL may offer structural advantages for adversarial robustness because the server can observe intermediate gradients and server-side behaviour during training. Among the defence approaches explored, Median/MAD-based filtering and SFL-aware layer-specific defences produced the strongest overall stability under extreme Non-IID conditions.

---

# 👤 Author

Raed Rahmanseresht
Master of Information Technology – UWA
