"""
End-to-end training loop for poisoning-resilient Split Federated Learning.

Execution flow (per FL round)
------------------------------
1. FL server selects a subset of clients.
2. Each selected client draws a local mini-batch, runs the front model
   forward pass, and sends *smashed data* + labels to the SL server.
3. The SL server completes the forward pass, computes the loss, and
   back-propagates.  It sends gradients back to the client.
4. The client updates its front model using the received gradients.
5. The SL server's multi-signal detector is updated with per-client
   signal values.  Suspected malicious clients are flagged / blacklisted.
6. After all selected clients have been processed, the FL server collects
   front-model state dicts, excludes blacklisted clients, and runs FedAvg.
7. The updated global front model is broadcast to all clients.
8. Every `log_interval` rounds, the global model is evaluated on the test
   set.

Usage
-----
  python main.py

Optional arguments can be changed by editing config.py or overriding
attributes on the `cfg` singleton before calling `run()`.
"""

import os
import random
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from client import Client
from config import Config, cfg
from data_utils import get_client_loaders, get_test_loader, load_cifar10
from fl_server import FLServer
from models import ClientModel
from sl_server import SLServer


# ---------------------------------------------------------------------- #
# Reproducibility                                                          #
# ---------------------------------------------------------------------- #

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------- #
# Client factory                                                           #
# ---------------------------------------------------------------------- #

def build_clients(cfg: Config) -> List[Client]:
    """Instantiate all clients, marking the malicious fraction."""
    num_malicious = int(cfg.num_clients * cfg.poison_fraction)
    # Assign the *last* num_malicious client IDs as adversarial.  Using a
    # fixed, deterministic assignment (rather than random selection) makes it
    # straightforward to compare detected blacklists against the ground truth
    # in experiments and unit tests.
    malicious_ids = set(range(cfg.num_clients - num_malicious, cfg.num_clients))

    clients = []
    for cid in range(cfg.num_clients):
        clients.append(
            Client(
                client_id=cid,
                cfg=cfg,
                is_malicious=(cid in malicious_ids),
            )
        )
    return clients


# ---------------------------------------------------------------------- #
# Evaluation                                                               #
# ---------------------------------------------------------------------- #

def evaluate_global_model(
    sl_server: SLServer,
    fl_server: FLServer,
    test_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate the combined (global front + server back) model on the test set.

    Returns (avg_loss, accuracy).
    """
    front_model = ClientModel(
        cut_layer=fl_server.cfg.cut_layer,
        num_classes=fl_server.cfg.num_classes,
    ).to(device)
    front_model.load_state_dict(fl_server.get_global_state())
    front_model.eval()
    sl_server.model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            smashed = front_model(images)
            loss, acc = sl_server.evaluate(smashed, labels)
            batch_size = labels.size(0)
            total_loss    += loss * batch_size
            total_correct += int(acc * batch_size)
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


# ---------------------------------------------------------------------- #
# Main training loop                                                       #
# ---------------------------------------------------------------------- #

def run(cfg: Config = cfg) -> Dict:
    """
    Run the full SFL training experiment and return a results dictionary.
    """
    set_seed(cfg.seed)
    os.makedirs(cfg.results_dir, exist_ok=True)
    device = torch.device(cfg.device)

    # ---- Data -------------------------------------------------------- #
    print("Loading CIFAR-10 …")
    train_dataset, test_dataset = load_cifar10(cfg.data_dir)
    client_loaders = get_client_loaders(cfg, train_dataset)
    test_loader    = get_test_loader(cfg, test_dataset)

    # ---- Models / servers -------------------------------------------- #
    sl_server = SLServer(cfg)
    fl_server = FLServer(cfg)
    clients   = build_clients(cfg)

    # Sync all clients with the initial global front model
    fl_server.broadcast(clients)

    malicious_ids = {c.client_id for c in clients if c.is_malicious}
    print(
        f"Clients: {cfg.num_clients} total, "
        f"{len(malicious_ids)} malicious {sorted(malicious_ids)}\n"
        f"Attack : {cfg.attack_type}\n"
        f"Non-IID: Dirichlet(alpha={cfg.dirichlet_alpha})\n"
    )

    # ---- Results tracking -------------------------------------------- #
    results = {
        "round":             [],
        "test_loss":         [],
        "test_acc":          [],
        "blacklisted":       [],
        "true_positive":     [],   # malicious clients correctly blacklisted
        "false_positive":    [],   # benign clients incorrectly blacklisted
        "round_time_s":      [],
    }

    # ---- Training rounds --------------------------------------------- #
    for round_num in range(cfg.num_rounds):
        t_start = time.time()

        sl_server.set_round(round_num)

        # Select clients for this round
        participating = random.sample(
            range(cfg.num_clients),
            min(cfg.clients_per_round, cfg.num_clients),
        )

        # Per-client loop
        client_states: Dict[int, dict] = {}
        round_losses: List[float] = []

        for cid in participating:
            client     = clients[cid]
            loader     = client_loaders[cid]
            loader_iter = iter(loader)

            # Run `local_epochs` worth of batches
            for _ in range(cfg.local_epochs):
                try:
                    images, labels = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(loader)
                    images, labels = next(loader_iter)

                # Client forward
                smashed, modified_labels = client.forward_pass(images, labels)

                # SL server forward + backward
                grad_smashed, batch_loss, _flagged = sl_server.process_client_batch(
                    client_id=cid,
                    smashed_data=smashed,
                    labels=modified_labels,
                )

                # Client backward
                client.backward_pass(grad_smashed)
                round_losses.append(batch_loss)

            client_states[cid] = client.get_model_state()

        # FL aggregation (exclude blacklisted clients)
        included = fl_server.aggregate(
            client_states=client_states,
            blacklist=sl_server.blacklisted_clients,
        )
        fl_server.broadcast(clients)

        t_end = time.time()

        # ---- Logging ------------------------------------------------- #
        blacklisted = sl_server.blacklisted_clients
        tp = len(blacklisted & malicious_ids)
        fp = len(blacklisted - malicious_ids)

        if (round_num + 1) % cfg.log_interval == 0 or round_num == 0:
            test_loss, test_acc = evaluate_global_model(
                sl_server, fl_server, test_loader, device
            )
            results["round"].append(round_num + 1)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)
            results["blacklisted"].append(sorted(blacklisted))
            results["true_positive"].append(tp)
            results["false_positive"].append(fp)
            results["round_time_s"].append(round(t_end - t_start, 2))

            avg_loss = sum(round_losses) / max(len(round_losses), 1)
            print(
                f"Round {round_num + 1:3d}/{cfg.num_rounds} | "
                f"train_loss={avg_loss:.4f}  "
                f"test_loss={test_loss:.4f}  "
                f"test_acc={test_acc:.3f}  "
                f"blacklisted={sorted(blacklisted)}  "
                f"TP={tp}  FP={fp}  "
                f"time={t_end - t_start:.1f}s"
            )

    # ---- Final summary ----------------------------------------------- #
    final_blacklisted = sl_server.blacklisted_clients
    final_tp = len(final_blacklisted & malicious_ids)
    final_fp = len(final_blacklisted - malicious_ids)
    total_malicious = len(malicious_ids)

    print("\n" + "=" * 60)
    print("Training complete.")
    print(f"Malicious clients  : {sorted(malicious_ids)}")
    print(f"Blacklisted clients: {sorted(final_blacklisted)}")
    print(f"True positives     : {final_tp}/{total_malicious}")
    print(f"False positives    : {final_fp}/{cfg.num_clients - total_malicious}")
    if total_malicious > 0:
        print(f"Detection rate     : {final_tp / total_malicious:.2%}")
    print("=" * 60)

    results["final_blacklisted"]  = sorted(final_blacklisted)
    results["final_true_positive"]  = final_tp
    results["final_false_positive"] = final_fp
    results["malicious_ids"]        = sorted(malicious_ids)

    return results


# ---------------------------------------------------------------------- #
# Entry point                                                              #
# ---------------------------------------------------------------------- #

if __name__ == "__main__":
    run(cfg)
