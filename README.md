# Poisoning-Resilient Split Federated Learning (SFL)

Masters project – Computer Science  
*SL-server-based detection of malicious clients using multi-signal training behaviour under non-IID data.*

---

## Overview

This repository implements a **Split Federated Learning (SFL)** framework that is resilient to poisoning attacks.
In SFL the neural network is split at a *cut layer*: each client runs the front layers locally and sends the intermediate activations (*smashed data*) to the SL server, which runs the remaining layers, computes the loss, and returns the gradients.
A separate FL aggregation server periodically collects client front-model weights and runs **FedAvg**.

The key contribution is a **multi-signal anomaly detector** built into the SL server.
Because the SL server observes *every client's* smashed data and gradients in every round, it is uniquely positioned to detect malicious behaviour without additional communication overhead.

---

## Architecture

```
                ┌─────────────────────────────────┐
                │         FL Aggregation Server    │
                │  FedAvg (excluding blacklisted)  │
                └────────────┬────────────────────┘
                             │ broadcast global front model
          ┌──────────────────┼──────────────────────────┐
          ▼                  ▼                           ▼
    ┌──────────┐       ┌──────────┐               ┌──────────┐
    │ Client 0 │       │ Client 1 │     …         │ Client N │
    │ (benign) │       │ (attack) │               │ (benign) │
    │ front    │       │ front    │               │ front    │
    │ model    │       │ model    │               │ model    │
    └────┬─────┘       └────┬─────┘               └────┬─────┘
         │ smashed data     │                           │
         └──────────────────┼───────────────────────────┘
                            ▼
                ┌───────────────────────┐
                │      SL Server        │
                │  back model           │
                │  + multi-signal       │
                │    detector           │
                └───────────────────────┘
```

---

## Poisoning Attacks Implemented

| Attack | Description |
|---|---|
| `label_flip` | All labels in the client's batch are replaced with a fixed target class. |
| `gradient_scale` | Gradients received from the SL server are scaled up by a configurable factor before the client update. |
| `backdoor` | A trigger pattern (small white square) is stamped on every image and labels are set to the backdoor target class. |

---

## Multi-Signal Detector

The SL server collects three signals per client per round:

| Signal | Description |
|---|---|
| `smash_norm` | L2 norm of the smashed-data tensor sent by the client. |
| `grad_norm` | L2 norm of the gradient tensor returned to the client. |
| `batch_loss` | Average cross-entropy loss for that client's batch. |

**Algorithm (per client per round)**:

1. Compute a per-signal Z-score against the client's *own* running mean/std (Welford's online algorithm).  Using per-client baselines makes the detector robust to non-IID data – clients are compared against themselves, not against a global average.
2. A client is **flagged** in round *t* if any signal Z-score exceeds `zscore_threshold`.
3. A client is **blacklisted** (excluded from FedAvg) after being flagged for `consecutive_flags` consecutive rounds.

An effective minimum standard deviation (`max(σ, 0.1·|μ| + ε)`) prevents both false negatives (std=0 from a constant baseline) and false positives (artificially small std from very low-noise signals).

Detection is skipped for the first `warmup_rounds` rounds so that each client's statistics can stabilise.

---

## Project Structure

```
.
├── config.py          # All hyperparameters (dataclass)
├── data_utils.py      # CIFAR-10 loading, Dirichlet non-IID split, backdoor trigger
├── models.py          # ClientModel (front), ServerModel (back), FullCNN
├── client.py          # Client: benign and malicious variants
├── detection.py       # _RunningStats, MaliciousClientDetector
├── sl_server.py       # SL server: forward/backward + detection integration
├── fl_server.py       # FL aggregation server: FedAvg + broadcast
├── main.py            # End-to-end training loop
├── requirements.txt
└── tests/
    └── test_sfl.py    # 40 unit + integration tests (pytest)
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run training

```bash
python main.py
```

This uses the default settings in `config.py`:
- 20 clients, 20 % malicious (`label_flip` attack)
- CIFAR-10 with Dirichlet non-IID splitting (α = 0.5)
- 50 FL rounds, 10 clients per round
- Detection enabled (Z-score threshold = 2.5, 2 consecutive flags to blacklist)

### 3. Run tests

```bash
python -m pytest tests/ -v
```

---

## Configuration

All settings live in `config.py` as a Python dataclass.  Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `num_clients` | 20 | Total number of federated clients |
| `poison_fraction` | 0.2 | Fraction of clients that are adversarial |
| `attack_type` | `"label_flip"` | `"label_flip"`, `"gradient_scale"`, or `"backdoor"` |
| `dirichlet_alpha` | 0.5 | Non-IID concentration (lower → more heterogeneous) |
| `cut_layer` | 2 | Split point (1–4 conv blocks on the client) |
| `num_rounds` | 50 | Number of global FL rounds |
| `detection_zscore_threshold` | 2.5 | Z-score above which a client is flagged |
| `detection_consecutive_flags` | 2 | Consecutive flags required for blacklisting |
| `detection_warmup_rounds` | 5 | Rounds before detection activates |

---

## Results Tracking

`main.py` prints per-round metrics and returns a dict with:

- `test_loss`, `test_acc` – global model performance on the CIFAR-10 test set
- `blacklisted` – client IDs excluded from aggregation at each logged round
- `true_positive` / `false_positive` – detection accuracy
- `malicious_ids` – ground-truth malicious client IDs

---

## Design Decisions

**Per-client baselines** – comparing a client against *itself* rather than the global population avoids false positives under extreme non-IID settings where benign clients with highly skewed data naturally produce unusual signal values.

**Conservative blacklisting** – requiring multiple *consecutive* anomalous rounds before excluding a client prevents one-off noisy rounds from unfairly removing benign clients.

**Effective minimum std** – the Z-score denominator is floored at `max(σ, 0.1|μ| + ε)`.  This handles the case where a client has been perfectly consistent historically (σ ≈ 0) and then suddenly exhibits drastically different behaviour.

**Modular architecture** – `SLServer`, `FLServer`, `Client`, and `MaliciousClientDetector` are independent classes that communicate through well-defined interfaces, making it straightforward to swap in alternative aggregation rules, attack strategies, or detection algorithms.
