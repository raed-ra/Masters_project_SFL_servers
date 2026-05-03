"""
Unit and integration tests for poisoning-resilient SFL.

Run with:
    python -m pytest tests/ -v
"""

import copy
import sys
import os

# Ensure the project root is on the path regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from client import Client
from config import Config
from data_utils import (
    dirichlet_partition,
    inject_backdoor_trigger,
)
from detection import MaliciousClientDetector, _RunningStats
from fl_server import FLServer
from models import ClientModel, FullCNN, ServerModel, count_parameters
from sl_server import SLServer


# ---------------------------------------------------------------------- #
# Fixtures                                                                 #
# ---------------------------------------------------------------------- #

@pytest.fixture
def base_cfg():
    cfg = Config()
    cfg.num_clients               = 10
    cfg.num_rounds                = 3
    cfg.clients_per_round         = 5
    cfg.poison_fraction           = 0.2
    cfg.cut_layer                 = 2
    cfg.local_epochs              = 1
    cfg.batch_size                = 8
    cfg.detection_warmup_rounds   = 1
    cfg.detection_consecutive_flags = 2
    cfg.detection_zscore_threshold  = 2.5
    cfg.device                    = "cpu"
    return cfg


def _fake_batch(batch_size=8, img_size=32, num_classes=10):
    """Return (images, labels) tensors for CIFAR-10-like data."""
    images = torch.randn(batch_size, 3, img_size, img_size)
    labels = torch.randint(0, num_classes, (batch_size,))
    return images, labels


# ---------------------------------------------------------------------- #
# models.py                                                                #
# ---------------------------------------------------------------------- #

class TestModels:

    @pytest.mark.parametrize("cut_layer", [1, 2, 3, 4])
    def test_split_forward_shape(self, cut_layer):
        """ClientModel + ServerModel forward pass produces correct logit shape."""
        client_model = ClientModel(cut_layer=cut_layer)
        server_model = ServerModel(cut_layer=cut_layer)

        x = torch.randn(4, 3, 32, 32)
        smashed = client_model(x)
        logits  = server_model(smashed)

        assert logits.shape == (4, 10), (
            f"Expected logit shape (4, 10), got {logits.shape}"
        )

    def test_full_cnn_forward(self):
        x = torch.randn(2, 3, 32, 32)
        model = FullCNN(num_classes=10)
        out = model(x)
        assert out.shape == (2, 10)

    def test_invalid_cut_layer_raises(self):
        with pytest.raises(ValueError):
            ClientModel(cut_layer=0)
        with pytest.raises(ValueError):
            ServerModel(cut_layer=5)

    def test_count_parameters_positive(self):
        model = FullCNN()
        assert count_parameters(model) > 0

    @pytest.mark.parametrize("cut_layer", [1, 2, 3])
    def test_parameter_split_adds_up(self, cut_layer):
        """Sum of client + server params should equal full model params."""
        full   = count_parameters(FullCNN())
        client = count_parameters(ClientModel(cut_layer=cut_layer))
        server = count_parameters(ServerModel(cut_layer=cut_layer))
        assert client + server == full, (
            f"cut_layer={cut_layer}: {client} + {server} != {full}"
        )


# ---------------------------------------------------------------------- #
# data_utils.py                                                            #
# ---------------------------------------------------------------------- #

class TestDataUtils:

    def _make_fake_dataset(self, n=200, num_classes=10):
        """Minimal stand-in for a torchvision dataset with `.targets`."""

        class FakeDataset:
            def __init__(self, n, nc):
                self.targets = list(np.random.randint(0, nc, n))
                self._data   = torch.randn(n, 3, 32, 32)
                self._labels = torch.tensor(self.targets)

            def __len__(self):
                return len(self.targets)

            def __getitem__(self, idx):
                return self._data[idx], self._labels[idx]

        return FakeDataset(n, num_classes)

    def test_dirichlet_partition_covers_all_indices(self):
        ds = self._make_fake_dataset(200)
        partition = dirichlet_partition(ds, num_clients=5, alpha=0.5, seed=0)
        all_indices = []
        for idx_list in partition.values():
            all_indices.extend(idx_list)
        assert sorted(all_indices) == list(range(200))

    def test_dirichlet_partition_num_clients(self):
        ds = self._make_fake_dataset(100)
        partition = dirichlet_partition(ds, num_clients=8, alpha=1.0, seed=1)
        assert len(partition) == 8

    def test_dirichlet_partition_non_iid(self):
        """Low alpha should produce more heterogeneous distributions."""
        ds = self._make_fake_dataset(500)
        partition_iid  = dirichlet_partition(ds, num_clients=5, alpha=100.0, seed=2)
        partition_niid = dirichlet_partition(ds, num_clients=5, alpha=0.1,   seed=2)

        def label_entropy(indices):
            labels = np.array(ds.targets)[indices]
            counts = np.bincount(labels, minlength=10).astype(float)
            counts = counts[counts > 0]
            p = counts / counts.sum()
            return float(-np.sum(p * np.log(p)))

        avg_iid  = np.mean([label_entropy(v) for v in partition_iid.values()])
        avg_niid = np.mean([label_entropy(v) for v in partition_niid.values()])
        assert avg_niid < avg_iid, "non-IID should have lower label entropy"

    def test_inject_backdoor_trigger_shape(self):
        images = torch.zeros(4, 3, 32, 32)
        out = inject_backdoor_trigger(images, trigger_size=4)
        assert out.shape == images.shape

    def test_inject_backdoor_trigger_values(self):
        images = torch.zeros(4, 3, 32, 32)
        trigger_size = 4
        out = inject_backdoor_trigger(images, trigger_size=trigger_size)
        assert (out[:, :, -trigger_size:, -trigger_size:] == 1.0).all()
        assert (out[:, :, :-trigger_size, :] == 0.0).all()

    def test_inject_backdoor_does_not_modify_original(self):
        images = torch.zeros(2, 3, 32, 32)
        _ = inject_backdoor_trigger(images, trigger_size=4)
        assert (images == 0).all(), "Original tensor should not be modified"


# ---------------------------------------------------------------------- #
# detection.py                                                             #
# ---------------------------------------------------------------------- #

class TestRunningStats:

    def test_single_value(self):
        rs = _RunningStats()
        rs.update(5.0)
        assert rs.mean == pytest.approx(5.0)
        assert rs.std  == pytest.approx(0.0)

    def test_mean_variance_multiple_values(self):
        rs = _RunningStats()
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            rs.update(v)
        assert rs.mean     == pytest.approx(3.0)
        assert rs.variance == pytest.approx(2.0)

    def test_zscore_extreme_value(self):
        rs = _RunningStats()
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            rs.update(v)
        assert rs.zscore(100.0) > 10


class TestDetector:

    def _make_detector(self, warmup=1, consecutive=2, threshold=2.5):
        return MaliciousClientDetector(
            signals=["smash_norm", "grad_norm", "batch_loss"],
            zscore_threshold=threshold,
            consecutive_flags=consecutive,
            warmup_rounds=warmup,
        )

    def test_warmup_no_flagging(self):
        det = self._make_detector(warmup=3)
        for r in range(3):
            flagged = det.update(
                0,
                smash_norm=1000.0, grad_norm=1000.0, batch_loss=1000.0,
                round_num=r,
            )
            assert not flagged

    def test_benign_client_not_blacklisted(self):
        rng = np.random.default_rng(42)
        det = self._make_detector(warmup=1, consecutive=2, threshold=2.5)
        for r in range(10):
            det.update(
                0,
                smash_norm=1.0  + rng.normal(0, 0.01),
                grad_norm =0.5  + rng.normal(0, 0.01),
                batch_loss=1.2  + rng.normal(0, 0.01),
                round_num=r,
            )
        assert not det.is_blacklisted(0)

    def test_malicious_client_eventually_blacklisted(self):
        det = self._make_detector(warmup=1, consecutive=2, threshold=2.0)
        for r in range(5):
            det.update(0, smash_norm=1.0, grad_norm=0.5, batch_loss=1.0,
                       round_num=r)
        for r in range(5, 10):
            det.update(0, smash_norm=100.0, grad_norm=100.0, batch_loss=100.0,
                       round_num=r)
        assert det.is_blacklisted(0)

    def test_blacklist_does_not_include_benign(self):
        det = self._make_detector(warmup=1, consecutive=2, threshold=2.5)
        for r in range(10):
            det.update(0, smash_norm=1.0, grad_norm=0.5, batch_loss=1.2,
                       round_num=r)
            det.update(
                1,
                smash_norm=100.0 if r > 2 else 1.0,
                grad_norm =100.0 if r > 2 else 0.5,
                batch_loss=100.0 if r > 2 else 1.2,
                round_num=r,
            )
        assert not det.is_blacklisted(0)

    def test_history_recorded(self):
        det = self._make_detector()
        for r in range(4):
            det.update(0, smash_norm=1.0, grad_norm=0.5, batch_loss=1.2,
                       round_num=r)
        history = det.get_history(0)
        assert len(history) == 4
        assert "round" in history[0]

    def test_summary_runs(self):
        det = self._make_detector()
        det.update(0, smash_norm=1.0, grad_norm=0.5, batch_loss=1.2,
                   round_num=2)
        assert isinstance(det.summary(), str)


# ---------------------------------------------------------------------- #
# client.py                                                                #
# ---------------------------------------------------------------------- #

class TestClient:

    def test_benign_forward_backward(self, base_cfg):
        client = Client(client_id=0, cfg=base_cfg, is_malicious=False)
        images, labels = _fake_batch(base_cfg.batch_size)
        smashed, returned_labels = client.forward_pass(images, labels)

        assert smashed.requires_grad
        assert returned_labels.shape == labels.shape

        grad = torch.randn_like(smashed)
        client.backward_pass(grad)

    def test_label_flip_attack(self, base_cfg):
        base_cfg.attack_type       = "label_flip"
        base_cfg.flip_target_label = 9
        client = Client(client_id=0, cfg=base_cfg, is_malicious=True)
        images, labels = _fake_batch(8)
        _, mod_labels = client.forward_pass(images, labels)
        assert (mod_labels == 9).all()

    def test_gradient_scale_no_error(self, base_cfg):
        base_cfg.attack_type            = "gradient_scale"
        base_cfg.gradient_scale_factor  = 3.0
        client = Client(client_id=0, cfg=base_cfg, is_malicious=True)
        images, labels = _fake_batch(8)
        smashed, _ = client.forward_pass(images, labels)
        client.backward_pass(torch.ones_like(smashed) * 2.0)

    def test_backdoor_attack(self, base_cfg):
        base_cfg.attack_type            = "backdoor"
        base_cfg.backdoor_target_label  = 0
        base_cfg.backdoor_trigger_size  = 4
        client = Client(client_id=0, cfg=base_cfg, is_malicious=True)
        images, labels = _fake_batch(8)
        _, mod_labels = client.forward_pass(images, labels)
        assert (mod_labels == 0).all()

    def test_get_set_model_state(self, base_cfg):
        c1 = Client(client_id=0, cfg=base_cfg)
        state = c1.get_model_state()
        c2 = Client(client_id=1, cfg=base_cfg)
        c2.set_model_state(state)
        for k in state:
            assert torch.allclose(
                c1.model.state_dict()[k],
                c2.model.state_dict()[k],
            )


# ---------------------------------------------------------------------- #
# sl_server.py                                                             #
# ---------------------------------------------------------------------- #

class TestSLServer:

    def _smashed(self, cfg):
        client_model = ClientModel(cut_layer=cfg.cut_layer)
        images, labels = _fake_batch(cfg.batch_size)
        with torch.no_grad():
            smashed = client_model(images)
        return smashed, labels

    def test_process_client_batch_shapes(self, base_cfg):
        sl = SLServer(base_cfg)
        smashed, labels = self._smashed(base_cfg)
        grad, loss, flagged = sl.process_client_batch(0, smashed, labels)
        assert grad.shape == smashed.shape
        assert isinstance(loss, float)
        assert isinstance(flagged, bool)

    def test_evaluate_returns_scalars(self, base_cfg):
        sl = SLServer(base_cfg)
        smashed, labels = self._smashed(base_cfg)
        loss, acc = sl.evaluate(smashed, labels)
        assert 0.0 <= acc <= 1.0
        assert loss >= 0.0

    def test_blacklist_initially_empty(self, base_cfg):
        sl = SLServer(base_cfg)
        assert len(sl.blacklisted_clients) == 0

    def test_detection_disabled(self, base_cfg):
        base_cfg.detection_enabled = False
        sl = SLServer(base_cfg)
        assert sl.detector is None
        smashed, labels = self._smashed(base_cfg)
        _, _, flagged = sl.process_client_batch(0, smashed, labels)
        assert not flagged


# ---------------------------------------------------------------------- #
# fl_server.py                                                             #
# ---------------------------------------------------------------------- #

class TestFLServer:

    def _state(self, cfg):
        return ClientModel(cut_layer=cfg.cut_layer).state_dict()

    def test_aggregate_uniform(self, base_cfg):
        fl = FLServer(base_cfg)
        states = {i: self._state(base_cfg) for i in range(3)}
        included = fl.aggregate(states)
        assert set(included) == {0, 1, 2}

    def test_aggregate_excludes_blacklist(self, base_cfg):
        fl = FLServer(base_cfg)
        states = {i: self._state(base_cfg) for i in range(5)}
        included = fl.aggregate(states, blacklist={1, 3})
        assert 1 not in included and 3 not in included
        assert 0 in included

    def test_aggregate_all_blacklisted(self, base_cfg):
        fl = FLServer(base_cfg)
        states = {i: self._state(base_cfg) for i in range(3)}
        old_state = copy.deepcopy(fl.global_model.state_dict())
        included = fl.aggregate(states, blacklist={0, 1, 2})
        assert included == []
        for k in old_state:
            assert torch.allclose(old_state[k], fl.global_model.state_dict()[k])

    def test_broadcast_updates_clients(self, base_cfg):
        fl = FLServer(base_cfg)
        clients = [Client(cid, base_cfg) for cid in range(3)]
        fl.broadcast(clients)
        gs = fl.get_global_state()
        for c in clients:
            for k in gs:
                assert torch.allclose(gs[k], c.model.state_dict()[k])

    def test_weighted_aggregation(self, base_cfg):
        fl = FLServer(base_cfg)
        m0 = ClientModel(cut_layer=base_cfg.cut_layer)
        m1 = ClientModel(cut_layer=base_cfg.cut_layer)
        for p in m0.parameters():
            p.data.fill_(0.0)
        for p in m1.parameters():
            p.data.fill_(2.0)
        fl.aggregate(
            {0: m0.state_dict(), 1: m1.state_dict()},
            client_weights={0: 1.0, 1: 1.0},
        )
        for p in fl.global_model.parameters():
            assert torch.allclose(p.data, torch.ones_like(p.data), atol=1e-5)


# ---------------------------------------------------------------------- #
# Integration                                                              #
# ---------------------------------------------------------------------- #

class TestIntegration:

    def test_mini_training_run(self, base_cfg):
        """Run 2 rounds with synthetic data; everything should wire together."""
        import random

        base_cfg.num_clients              = 6
        base_cfg.clients_per_round        = 3
        base_cfg.poison_fraction          = 1 / 6
        base_cfg.num_rounds               = 2
        base_cfg.detection_warmup_rounds  = 1
        base_cfg.detection_consecutive_flags = 2

        sl = SLServer(base_cfg)
        fl = FLServer(base_cfg)

        malicious_ids = {5}
        clients = [
            Client(cid, base_cfg, is_malicious=(cid in malicious_ids))
            for cid in range(base_cfg.num_clients)
        ]
        fl.broadcast(clients)

        for round_num in range(base_cfg.num_rounds):
            sl.set_round(round_num)
            participating = random.sample(
                range(base_cfg.num_clients), base_cfg.clients_per_round
            )
            client_states = {}
            for cid in participating:
                images, labels = _fake_batch(base_cfg.batch_size)
                smashed, mod_labels = clients[cid].forward_pass(images, labels)
                grad, loss, _ = sl.process_client_batch(cid, smashed, mod_labels)
                clients[cid].backward_pass(grad)
                client_states[cid] = clients[cid].get_model_state()

            fl.aggregate(client_states, blacklist=sl.blacklisted_clients)
            fl.broadcast(clients)

        assert fl.global_model is not None
