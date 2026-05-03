"""
Data utilities for poisoning-resilient SFL.

Provides:
  - CIFAR-10 loading with standard augmentation.
  - Dirichlet-based non-IID partitioning for federated clients.
  - Per-client DataLoader factory.
"""

import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from config import Config


# ---------------------------------------------------------------------- #
# Transforms                                                               #
# ---------------------------------------------------------------------- #

def _get_cifar10_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010),
            ),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010),
        ),
    ])


# ---------------------------------------------------------------------- #
# Dataset loading                                                          #
# ---------------------------------------------------------------------- #

def load_cifar10(
    data_dir: str,
) -> Tuple[datasets.CIFAR10, datasets.CIFAR10]:
    """Return (train_dataset, test_dataset) for CIFAR-10."""
    train_ds = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=_get_cifar10_transforms(train=True),
    )
    test_ds = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=_get_cifar10_transforms(train=False),
    )
    return train_ds, test_ds


# ---------------------------------------------------------------------- #
# Non-IID partitioning (Dirichlet)                                         #
# ---------------------------------------------------------------------- #

def dirichlet_partition(
    dataset: datasets.VisionDataset,
    num_clients: int,
    alpha: float,
    seed: int = 42,
) -> Dict[int, List[int]]:
    """
    Partition dataset indices across `num_clients` clients using a
    Dirichlet(alpha) distribution over class labels.

    Parameters
    ----------
    dataset    : torchvision dataset (must expose `.targets`).
    num_clients: total number of clients.
    alpha      : concentration parameter.  Small values give highly
                 heterogeneous splits; alpha → ∞ approaches IID.
    seed       : random seed for reproducibility.

    Returns
    -------
    Dict mapping client_id → list of sample indices.
    """
    rng = np.random.default_rng(seed)
    labels = np.array(dataset.targets)
    num_classes = len(np.unique(labels))

    # Collect per-class indices
    class_indices: Dict[int, np.ndarray] = {
        c: np.where(labels == c)[0] for c in range(num_classes)
    }

    client_indices: Dict[int, List[int]] = {i: [] for i in range(num_clients)}

    for cls, idx in class_indices.items():
        rng.shuffle(idx)
        # Sample proportions from Dirichlet
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        # Convert proportions to cumulative split points
        splits = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        for client_id, chunk in enumerate(np.split(idx, splits)):
            client_indices[client_id].extend(chunk.tolist())

    # Shuffle each client's indices
    for cid in client_indices:
        random.Random(seed + cid).shuffle(client_indices[cid])

    return client_indices


# ---------------------------------------------------------------------- #
# DataLoader factory                                                       #
# ---------------------------------------------------------------------- #

def get_client_loaders(
    cfg: Config,
    train_dataset: datasets.VisionDataset,
) -> Dict[int, DataLoader]:
    """
    Build a DataLoader for every client using the Dirichlet partition.

    Returns
    -------
    Dict mapping client_id → DataLoader.
    """
    partition = dirichlet_partition(
        train_dataset,
        num_clients=cfg.num_clients,
        alpha=cfg.dirichlet_alpha,
        seed=cfg.seed,
    )
    loaders: Dict[int, DataLoader] = {}
    for cid, indices in partition.items():
        subset = Subset(train_dataset, indices)
        loaders[cid] = DataLoader(
            subset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )
    return loaders


def get_test_loader(cfg: Config, test_dataset: datasets.VisionDataset) -> DataLoader:
    """Return a single test DataLoader for global evaluation."""
    return DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
    )


# ---------------------------------------------------------------------- #
# Backdoor trigger injection                                               #
# ---------------------------------------------------------------------- #

def inject_backdoor_trigger(
    images: torch.Tensor,
    trigger_size: int,
    value: float = 1.0,
) -> torch.Tensor:
    """
    Stamp a small white square in the bottom-right corner of every image.

    Parameters
    ----------
    images      : (B, C, H, W) float tensor.
    trigger_size: side length of the trigger square in pixels.
    value       : pixel value to write (post-normalisation, so 1.0 is
                  approximately correct for CIFAR-10).

    Returns
    -------
    Modified image tensor (same shape, cloned).
    """
    imgs = images.clone()
    h, w = imgs.shape[-2], imgs.shape[-1]
    imgs[:, :, h - trigger_size:h, w - trigger_size:w] = value
    return imgs
