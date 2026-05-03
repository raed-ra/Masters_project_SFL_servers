"""
Client implementation for poisoning-resilient SFL.

Each client:
  1. Holds a *front* model (ClientModel).
  2. Runs a local forward pass to produce *smashed data*.
  3. Receives gradients from the SL server and back-propagates through
     its front model.
  4. May be *malicious* (label-flip, gradient-scale, or backdoor attacker).

Usage
-----
  client = Client(client_id=0, cfg=cfg, is_malicious=False)
  smashed, labels = client.forward_pass(images, labels)
  client.backward_pass(grad_smashed)
  state_dict = client.get_model_state()
  client.set_model_state(state_dict)
"""

import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from config import Config
from data_utils import inject_backdoor_trigger
from models import ClientModel


class Client:
    """
    Represents a single federated client in an SFL system.

    Parameters
    ----------
    client_id    : unique integer identifier.
    cfg          : global Config object.
    is_malicious : if True the client will mount an attack defined by
                   cfg.attack_type.
    """

    def __init__(
        self,
        client_id: int,
        cfg: Config,
        is_malicious: bool = False,
    ) -> None:
        self.client_id    = client_id
        self.cfg          = cfg
        self.is_malicious = is_malicious
        self.device       = torch.device(cfg.device)

        self.model = ClientModel(
            cut_layer=cfg.cut_layer,
            num_classes=cfg.num_classes,
        ).to(self.device)

        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=cfg.learning_rate,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )

        # Stored for backward pass
        self._smashed_data: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ #
    # Forward pass                                                         #
    # ------------------------------------------------------------------ #

    def forward_pass(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run the front model and optionally apply the chosen attack.

        Returns
        -------
        smashed_data : detached tensor that will be sent to the SL server.
        labels       : (possibly modified) label tensor.
        """
        images = images.to(self.device)
        labels = labels.to(self.device)

        # ---- Poisoning ------------------------------------------------ #
        if self.is_malicious:
            images, labels = self._apply_attack(images, labels)

        # ---- Front-model forward -------------------------------------- #
        self.model.train()
        self.optimizer.zero_grad()

        # Keep computation graph so we can back-prop later
        smashed = self.model(images)

        # Detach before sending to server (prevents full back-prop leak)
        # We keep a reference with grad enabled for the backward step
        self._smashed_data = smashed

        return smashed.detach().requires_grad_(True), labels

    # ------------------------------------------------------------------ #
    # Backward pass                                                        #
    # ------------------------------------------------------------------ #

    def backward_pass(self, grad_smashed: torch.Tensor) -> None:
        """
        Receive gradients from the SL server and update the front model.

        Parameters
        ----------
        grad_smashed : gradient tensor w.r.t. the smashed data, same shape
                       as the tensor returned by `forward_pass`.
        """
        if self._smashed_data is None:
            raise RuntimeError("Call forward_pass before backward_pass.")

        grad_smashed = grad_smashed.to(self.device)

        # Apply gradient scaling attack (amplify influence)
        if self.is_malicious and self.cfg.attack_type == "gradient_scale":
            grad_smashed = grad_smashed * self.cfg.gradient_scale_factor

        self._smashed_data.backward(grad_smashed)
        self.optimizer.step()
        self._smashed_data = None

    # ------------------------------------------------------------------ #
    # Model state                                                          #
    # ------------------------------------------------------------------ #

    def get_model_state(self) -> dict:
        """Return a deep copy of the front-model state dict."""
        return copy.deepcopy(self.model.state_dict())

    def set_model_state(self, state_dict: dict) -> None:
        """Load a front-model state dict (e.g. from FL aggregation)."""
        self.model.load_state_dict(copy.deepcopy(state_dict))

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _apply_attack(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the poisoning attack configured in cfg."""
        attack = self.cfg.attack_type

        if attack == "label_flip":
            # Flip all labels to a fixed target class
            labels = torch.full_like(labels, self.cfg.flip_target_label)

        elif attack == "backdoor":
            # Stamp trigger and re-label as backdoor target
            images = inject_backdoor_trigger(
                images,
                trigger_size=self.cfg.backdoor_trigger_size,
            )
            labels = torch.full_like(labels, self.cfg.backdoor_target_label)

        elif attack == "gradient_scale":
            # No image/label modification; scaling happens in backward_pass
            pass

        else:
            raise ValueError(f"Unknown attack type: {attack!r}")

        return images, labels
