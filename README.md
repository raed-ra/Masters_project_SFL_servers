# Model Poisoning-Resilient Split Federated Learning (SFL)

## 📌 Overview
This project investigates Federated Learning (FL), Split Learning (SL), and their combination (SFL), with the ultimate goal of developing a framework robust against poisoning attacks.

The work progresses from baseline FL implementations to Split Learning and controlled comparisons, forming the foundation for a poisoning-resilient SFL system.

---

## 🎯 Objectives
- Implement baseline Federated Learning
- Extend experiments to multiple datasets
- Implement Split Learning architecture
- Compare FL and SL under identical conditions (Design 1)
- Prepare groundwork for poisoning attack simulation and defence mechanisms

---

## 📁 Project Structure



notebooks/
│ 01_fl_baseline_mnist.ipynb
│ 02_fl_multidataset_analysis.ipynb
│ 03_split_learning_implementation.ipynb
│ 04_fl_vs_sl_equivalence_study.ipynb

results/
│ plots/
│ csv/


---

## 🧠 Notebook Summaries

### 1. FL Baseline (MNIST)
- Implements standard Federated Learning using MNIST
- Uses FedAvg aggregation
- Tracks accuracy and loss per round

**Purpose:** Establish baseline distributed learning performance

---

### 2. FL Multi-Dataset Analysis
- Runs FL on MNIST, Fashion-MNIST, and CIFAR-10
- Evaluates performance across datasets of increasing complexity

**Key Insight:** Performance degrades as dataset complexity increases

---

### 3. Split Learning Implementation
- Splits model into client-side and server-side
- Clients send activations instead of weights
- Server performs forward/backward propagation

**Purpose:** Enable learning for resource-constrained clients

---

### 4. FL vs SL Equivalence Study
- Runs FL and SL under identical conditions
- Compares accuracy and model similarity (L2 distance)

**Key Insight:** SL can approximate FL under controlled conditions

---

## 📊 Results Summary

### 🔹 FL Performance
- Stable convergence on MNIST
- Performance drops with non-IID data and complex datasets

### 🔹 SL Behaviour
- Comparable to FL under controlled setup
- More sensitive to training dynamics

### 🔹 FL vs SL Comparison
- Similar behaviour when initialized identically
- Divergence highlights structural differences

---

## ⚠️ Key Observations
- Non-IID data significantly impacts distributed learning
- SL introduces new attack surfaces:
  - Activations
  - Gradients
- FL aggregation alone is insufficient for security

---

## 🚀 Next Steps
- Implement poisoning attacks:
  - Label flipping
  - Model poisoning
  - Gradient manipulation
  - Activation manipulation
- Develop SL-server-based detection:
  - Activation distribution analysis
  - Gradient stability tracking
  - Loss trajectory monitoring

---

## 🧠 Conclusion
This project demonstrates that while Split Learning can replicate Federated Learning behaviour under controlled conditions, it introduces additional vulnerabilities.

Future work focuses on developing robust detection mechanisms at the SL server.

---

## 👤 Author
Raed Rahmanseresht  
Master of Information Technology – UWA