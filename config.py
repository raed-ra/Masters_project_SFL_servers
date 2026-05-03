"""
Configuration for poisoning-resilient Split Federated Learning (SFL).

Architecture
------------
- Clients      : hold the *front* portion of the model (up to the cut layer).
- SL Server    : holds the *back* portion.  Receives smashed data, runs the
                 rest of the forward pass, computes the loss, and returns
                 gradients.  Also runs the multi-signal malicious-client
                 detector.
- FL Server    : aggregates client front-model weights (FedAvg), optionally
                 excluding clients flagged as malicious.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ------------------------------------------------------------------ #
    # Data                                                                 #
    # ------------------------------------------------------------------ #
    dataset: str = "CIFAR10"          # only CIFAR-10 is currently wired up
    data_dir: str = "./data"
    num_classes: int = 10

    # Non-IID data distribution
    # alpha controls the concentration of the Dirichlet distribution.
    # Smaller alpha → more heterogeneous (extreme non-IID).
    dirichlet_alpha: float = 0.5

    # ------------------------------------------------------------------ #
    # Federated / split setup                                              #
    # ------------------------------------------------------------------ #
    num_clients: int = 20
    num_rounds: int = 50              # global FL rounds
    clients_per_round: int = 10       # how many clients participate each round

    # Fraction of clients that are adversarial
    poison_fraction: float = 0.2      # 0.2 → 20 % of total clients are bad
    # Type of poisoning attack: "label_flip" | "gradient_scale" | "backdoor"
    attack_type: str = "label_flip"
    # Label-flip target class (used when attack_type == "label_flip")
    flip_target_label: int = 0
    # Gradient scale factor (used when attack_type == "gradient_scale")
    gradient_scale_factor: float = 5.0
    # Backdoor trigger size in pixels (used when attack_type == "backdoor")
    backdoor_trigger_size: int = 4
    # Backdoor target class
    backdoor_target_label: int = 0

    # ------------------------------------------------------------------ #
    # Model split                                                          #
    # ------------------------------------------------------------------ #
    # Number of convolutional blocks kept on the client side.
    # The remaining blocks run on the SL server.
    cut_layer: int = 2                # split after the 2nd conv block

    # ------------------------------------------------------------------ #
    # Training hyper-parameters                                            #
    # ------------------------------------------------------------------ #
    local_epochs: int = 1             # local steps per round (per client)
    batch_size: int = 64
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4

    # ------------------------------------------------------------------ #
    # Multi-signal detection                                               #
    # ------------------------------------------------------------------ #
    # Signals collected by the SL server per client per round:
    #   - smash_norm   : L2 norm of the smashed-data tensor
    #   - smash_mean   : element-wise mean of smashed data
    #   - smash_var    : element-wise variance of smashed data
    #   - grad_norm    : L2 norm of the gradient sent back to the client
    #   - batch_loss   : average cross-entropy loss for this client's batch
    #
    # Detection is triggered after `detection_warmup_rounds` rounds so
    # that per-client statistics have time to stabilise.
    detection_enabled: bool = True
    detection_warmup_rounds: int = 5
    # Z-score threshold: clients whose signal z-score exceeds this value
    # are flagged as suspicious.
    detection_zscore_threshold: float = 2.5
    # Number of consecutive rounds a client must be flagged before being
    # excluded from aggregation.
    detection_consecutive_flags: int = 2
    # Signals used for detection (subset of those described above)
    detection_signals: List[str] = field(
        default_factory=lambda: [
            "smash_norm",
            "grad_norm",
            "batch_loss",
        ]
    )

    # ------------------------------------------------------------------ #
    # Misc                                                                 #
    # ------------------------------------------------------------------ #
    seed: int = 42
    device: str = "cpu"               # "cuda" if GPU is available
    log_interval: int = 5             # print metrics every N rounds
    results_dir: str = "./results"


# Convenience singleton used throughout the project
cfg = Config()
