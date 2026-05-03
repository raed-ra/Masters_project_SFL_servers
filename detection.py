"""
Multi-signal malicious-client detector for the SL server.

Design goals
------------
* **Non-IID robustness**: the detector tracks *per-client* running
  statistics, so it compares each client against its own historical
  baseline rather than against a global mean.  This avoids false
  positives caused by heterogeneous data distributions.
* **Multi-signal fusion**: three signals are aggregated – smashed-data
  L2 norm, gradient L2 norm, and per-batch cross-entropy loss.
* **Conservative flagging**: a client is only *excluded* from
  aggregation after being flagged for `consecutive_flags` consecutive
  rounds.

Algorithm (per round, per client)
----------------------------------
1. Collect the three signal values for the current round.
2. Compute a Z-score for each signal relative to the client's own
   running (mean, std) computed over all past rounds.
3. A client is **flagged** in round t if ANY single signal Z-score
   exceeds `zscore_threshold`.
4. A client is **blacklisted** (excluded from FedAvg) if it has been
   flagged for `consecutive_flags` rounds in a row.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------- #
# Per-client running statistics                                            #
# ---------------------------------------------------------------------- #

class _RunningStats:
    """Welford's online algorithm for mean and variance."""

    def __init__(self) -> None:
        self.n: int       = 0
        self.mean: float  = 0.0
        self.M2: float    = 0.0   # sum of squared deviations

    def update(self, value: float) -> None:
        self.n += 1
        delta  = value - self.mean
        self.mean += delta / self.n
        self.M2   += delta * (value - self.mean)

    @property
    def variance(self) -> float:
        return self.M2 / self.n if self.n > 1 else 0.0

    @property
    def std(self) -> float:
        return float(np.sqrt(self.variance))

    def zscore(self, value: float) -> float:
        """
        Return |Z| for `value` given current statistics.

        When the running standard deviation is very small (e.g. because all
        historical observations were identical), a pure z-score would return
        0.0 even for wildly different new values.  To avoid missing large
        absolute deviations we use an *effective* std that is at least 10 %
        of the running mean (plus a small absolute floor).  This keeps
        benign clients with naturally stable signals well below the threshold
        while still detecting sudden large jumps.
        """
        if self.n < 2:
            return 0.0
        # Effective std: never less than 10 % of |mean| + small absolute floor
        effective_std = max(self.std, 0.10 * abs(self.mean) + 1e-4)
        return abs((value - self.mean) / effective_std)


# ---------------------------------------------------------------------- #
# Detector                                                                 #
# ---------------------------------------------------------------------- #

class MaliciousClientDetector:
    """
    SL-server-side multi-signal anomaly detector.

    Parameters
    ----------
    signals              : list of signal names to use.
    zscore_threshold     : flag a client if any signal Z-score exceeds
                           this value.
    consecutive_flags    : number of consecutive flagged rounds required
                           before the client is blacklisted.
    warmup_rounds        : skip detection during the first N rounds so
                           that per-client statistics can stabilise.
    """

    def __init__(
        self,
        signals: List[str],
        zscore_threshold: float = 2.5,
        consecutive_flags: int = 2,
        warmup_rounds: int = 5,
    ) -> None:
        self.signals            = signals
        self.zscore_threshold   = zscore_threshold
        self.consecutive_flags  = consecutive_flags
        self.warmup_rounds      = warmup_rounds

        # Per-client, per-signal running stats
        # stats[client_id][signal_name] → _RunningStats
        self._stats: Dict[int, Dict[str, _RunningStats]] = defaultdict(
            lambda: {s: _RunningStats() for s in signals}
        )

        # Consecutive-flag counter per client
        self._consec: Dict[int, int] = defaultdict(int)

        # Set of blacklisted client IDs
        self._blacklist: Set[int] = set()

        # Full history for post-hoc analysis / plotting
        # history[client_id] → list of dicts
        self._history: Dict[int, List[dict]] = defaultdict(list)

        self._current_round: int = 0

    # ------------------------------------------------------------------ #
    # Main interface                                                       #
    # ------------------------------------------------------------------ #

    def update(
        self,
        client_id: int,
        smash_norm: float,
        grad_norm: float,
        batch_loss: float,
        round_num: int,
    ) -> bool:
        """
        Record signals for one client in one round and return whether the
        client is *flagged* this round (not necessarily blacklisted).

        Parameters
        ----------
        client_id  : integer client identifier.
        smash_norm : L2 norm of the smashed data sent by this client.
        grad_norm  : L2 norm of the gradient sent back to this client.
        batch_loss : average cross-entropy loss for this client's batch.
        round_num  : current global round index (0-based).

        Returns
        -------
        True if the client is flagged this round; False otherwise.
        """
        self._current_round = round_num
        values = {
            "smash_norm": smash_norm,
            "grad_norm":  grad_norm,
            "batch_loss": batch_loss,
        }

        # Collect only the configured signals
        signal_values = {s: values[s] for s in self.signals if s in values}

        # Compute Z-scores BEFORE updating running stats so the new
        # observation does not influence its own Z-score.
        zscores: Dict[str, float] = {}
        for sig, val in signal_values.items():
            zscores[sig] = self._stats[client_id][sig].zscore(val)

        # Update running stats
        for sig, val in signal_values.items():
            self._stats[client_id][sig].update(val)

        # Record history (signal values + their z-scores for this round)
        zscore_entries = {f"z_{s}": z for s, z in zscores.items()}
        record = {"round": round_num, **signal_values, **zscore_entries}
        self._history[client_id].append(record)

        # Warmup: skip detection decision
        if round_num < self.warmup_rounds:
            return False

        # Flag if any signal Z-score exceeds the threshold
        flagged = any(z > self.zscore_threshold for z in zscores.values())

        if flagged:
            self._consec[client_id] += 1
        else:
            self._consec[client_id] = 0  # reset on clean round

        # Blacklist after enough consecutive flags
        if self._consec[client_id] >= self.consecutive_flags:
            self._blacklist.add(client_id)

        return flagged

    # ------------------------------------------------------------------ #
    # Queries                                                              #
    # ------------------------------------------------------------------ #

    def is_blacklisted(self, client_id: int) -> bool:
        return client_id in self._blacklist

    @property
    def blacklisted_clients(self) -> Set[int]:
        return set(self._blacklist)

    def get_history(self, client_id: int) -> List[dict]:
        return list(self._history[client_id])

    def get_all_history(self) -> Dict[int, List[dict]]:
        return {cid: list(records) for cid, records in self._history.items()}

    def summary(self) -> str:
        lines = [
            f"Round {self._current_round} | "
            f"Blacklisted clients: {sorted(self._blacklist)}"
        ]
        for cid in sorted(self._history.keys()):
            consec = self._consec[cid]
            bl     = "BLACKLISTED" if cid in self._blacklist else "ok"
            lines.append(
                f"  client {cid:3d}: consecutive_flags={consec}  status={bl}"
            )
        return "\n".join(lines)
