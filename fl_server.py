"""
Federated Learning (FL) aggregation server for poisoning-resilient SFL.

Responsibilities
----------------
1. Collect front-model state dicts from participating clients.
2. Run **FedAvg** aggregation, optionally skipping clients that the SL
   server has blacklisted.
3. Broadcast the aggregated global front-model back to all clients.

This module is intentionally thin: all detection logic lives in
`detection.py` / `sl_server.py`.
"""

import copy
from typing import Dict, List, Optional, Set

import torch

from config import Config
from models import ClientModel


class FLServer:
    """
    Federated Learning aggregation server.

    Parameters
    ----------
    cfg : global Config object.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg    = cfg
        self.device = torch.device(cfg.device)

        # Global front-model (initialised once; distributed to clients)
        self.global_model = ClientModel(
            cut_layer=cfg.cut_layer,
            num_classes=cfg.num_classes,
        ).to(self.device)

    # ------------------------------------------------------------------ #
    # Core interface                                                       #
    # ------------------------------------------------------------------ #

    def aggregate(
        self,
        client_states: Dict[int, dict],
        blacklist: Optional[Set[int]] = None,
        client_weights: Optional[Dict[int, float]] = None,
    ) -> List[int]:
        """
        Run FedAvg over client front-model state dicts.

        Parameters
        ----------
        client_states  : mapping {client_id: state_dict}.
        blacklist      : set of client IDs to exclude from aggregation.
        client_weights : optional per-client weighting (e.g. local dataset
                         size).  If None, uniform weighting is used.

        Returns
        -------
        List of client IDs that were *included* in aggregation.
        """
        blacklist = blacklist or set()

        included = {
            cid: state
            for cid, state in client_states.items()
            if cid not in blacklist
        }

        if not included:
            # No safe clients this round – keep the global model unchanged
            return []

        # Determine weights
        if client_weights is not None:
            weights = {
                cid: client_weights[cid]
                for cid in included
                if cid in client_weights
            }
        else:
            weights = {cid: 1.0 for cid in included}

        total_weight = sum(weights.values())
        if total_weight == 0:
            return []

        # Weighted average of state dicts
        avg_state: dict = {}
        for cid, state in included.items():
            w = weights.get(cid, 1.0) / total_weight
            for key, param in state.items():
                p = param.float().to(self.device)
                if key not in avg_state:
                    avg_state[key] = p * w
                else:
                    avg_state[key] += p * w

        self.global_model.load_state_dict(avg_state)
        return list(included.keys())

    # ------------------------------------------------------------------ #
    # Distribution                                                         #
    # ------------------------------------------------------------------ #

    def get_global_state(self) -> dict:
        """Return a copy of the global front-model state dict."""
        return copy.deepcopy(self.global_model.state_dict())

    def broadcast(self, clients: list) -> None:
        """
        Push the global front-model to every client in `clients`.

        Parameters
        ----------
        clients : list of Client objects (must have a `set_model_state`
                  method).
        """
        global_state = self.get_global_state()
        for client in clients:
            client.set_model_state(global_state)
