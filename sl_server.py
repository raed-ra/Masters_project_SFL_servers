"""
Split Learning (SL) server for poisoning-resilient SFL.

Responsibilities
----------------
1. Hold and train the *back* portion of the model (ServerModel).
2. Receive smashed data from clients, complete the forward pass, compute
   the cross-entropy loss, and back-propagate.
3. Return the gradient w.r.t. the smashed data to the client.
4. Feed per-client signal values to the MaliciousClientDetector after
   each client batch.
5. Expose the gradient tensor so the FL server / main loop can relay it.

The SL server does NOT aggregate client front-model weights – that is the
FL server's job (see fl_server.py).
"""

from typing import Dict, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from config import Config
from detection import MaliciousClientDetector
from models import ServerModel


class SLServer:
    """
    Split Learning server.

    Parameters
    ----------
    cfg       : global Config object.
    detector  : optional MaliciousClientDetector instance.  If None and
                cfg.detection_enabled is True, one is created internally.
    """

    def __init__(
        self,
        cfg: Config,
        detector: Optional[MaliciousClientDetector] = None,
    ) -> None:
        self.cfg    = cfg
        self.device = torch.device(cfg.device)

        self.model = ServerModel(
            cut_layer=cfg.cut_layer,
            num_classes=cfg.num_classes,
        ).to(self.device)

        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=cfg.learning_rate,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
        self.criterion = nn.CrossEntropyLoss()

        # Detection
        if cfg.detection_enabled:
            self.detector: Optional[MaliciousClientDetector] = (
                detector
                if detector is not None
                else MaliciousClientDetector(
                    signals=cfg.detection_signals,
                    zscore_threshold=cfg.detection_zscore_threshold,
                    consecutive_flags=cfg.detection_consecutive_flags,
                    warmup_rounds=cfg.detection_warmup_rounds,
                )
            )
        else:
            self.detector = None

        # Round counter (incremented externally via set_round)
        self._round: int = 0

    # ------------------------------------------------------------------ #
    # Round management                                                     #
    # ------------------------------------------------------------------ #

    def set_round(self, round_num: int) -> None:
        """Notify the server of the current global round index."""
        self._round = round_num

    # ------------------------------------------------------------------ #
    # Core computation                                                     #
    # ------------------------------------------------------------------ #

    def process_client_batch(
        self,
        client_id: int,
        smashed_data: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, float, bool]:
        """
        Run one forward–backward cycle for a single client's batch.

        Parameters
        ----------
        client_id    : integer identifier of the calling client.
        smashed_data : detached activation tensor from the client's front
                       model – shape (B, C, H', W').
        labels       : ground-truth (or poisoned) labels – shape (B,).

        Returns
        -------
        grad_smashed : gradient tensor w.r.t. smashed_data (same shape),
                       to be sent back to the client.
        batch_loss   : scalar loss value for logging / detection.
        flagged      : True if the detector flagged this client this round.
        """
        smashed_data = smashed_data.to(self.device)
        labels       = labels.to(self.device)

        # Re-enable gradients on the received tensor
        smashed_data = smashed_data.detach().requires_grad_(True)

        # Forward pass through server model
        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model(smashed_data)
        loss   = self.criterion(logits, labels)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        # Gradient to return to client
        assert smashed_data.grad is not None, (
            "Gradient w.r.t. smashed_data is None – check model structure."
        )
        grad_smashed = smashed_data.grad.detach().clone()

        batch_loss_val = float(loss.item())

        # Signal collection for detection
        smash_norm = float(smashed_data.detach().norm().item())
        grad_norm  = float(grad_smashed.norm().item())

        flagged = False
        if self.detector is not None:
            flagged = self.detector.update(
                client_id  = client_id,
                smash_norm = smash_norm,
                grad_norm  = grad_norm,
                batch_loss = batch_loss_val,
                round_num  = self._round,
            )

        return grad_smashed, batch_loss_val, flagged

    # ------------------------------------------------------------------ #
    # Evaluation (no gradient tracking)                                    #
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        smashed_data: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[float, float]:
        """
        Compute loss and top-1 accuracy for a batch.

        Returns
        -------
        (loss, accuracy) as Python floats.
        """
        smashed_data = smashed_data.to(self.device)
        labels       = labels.to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits  = self.model(smashed_data)
            loss    = self.criterion(logits, labels)
            preds   = logits.argmax(dim=1)
            correct = (preds == labels).sum().item()
            acc     = correct / labels.size(0)
        return float(loss.item()), acc

    # ------------------------------------------------------------------ #
    # Blacklist query                                                      #
    # ------------------------------------------------------------------ #

    @property
    def blacklisted_clients(self) -> Set[int]:
        if self.detector is None:
            return set()
        return self.detector.blacklisted_clients

    def is_blacklisted(self, client_id: int) -> bool:
        if self.detector is None:
            return False
        return self.detector.is_blacklisted(client_id)

    # ------------------------------------------------------------------ #
    # State dict (for checkpointing)                                       #
    # ------------------------------------------------------------------ #

    def get_model_state(self) -> dict:
        return self.model.state_dict()

    def set_model_state(self, state_dict: dict) -> None:
        self.model.load_state_dict(state_dict)
