"""Solved linear readout: fitted Q by least squares (no CMA-ES search noise).

The original design evolved a linear readout with CMA-ES, which is exactly the step the
repo's own data showed to be search-limited (candidate reliability ~0.006). A *linear*
readout does not need a search: with a fixed memory, the optimal one-step Q is a least
squares problem. We fit it by fitted Q-iteration on recorded transitions

    Q(s, a) = r + gamma * max_a' Q(s', a')

and act greedily. The recursion is the standard FQI fixed point; with a linear function
class each iteration is one ridge solve, so it is cheap and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._types import F32


@dataclass(frozen=True)
class LinearQ:
    W: F32  # (n_features + 1, n_actions); last row is the bias

    @staticmethod
    def fit(
        S: F32,
        A: np.ndarray,
        R: F32,
        S2: F32,
        D: np.ndarray,
        n_actions: int,
        gamma: float = 0.99,
        ridge: float = 1e-3,
        iters: int = 12,
    ) -> LinearQ:
        """Fitted Q-iteration on transitions ``(S, A, R, S2, D)`` (D = 1 if terminal)."""
        X = np.hstack([S, np.ones((len(S), 1), np.float32)]).astype(np.float32)
        X2 = np.hstack([S2, np.ones((len(S2), 1), np.float32)]).astype(np.float32)
        A = np.asarray(A, dtype=np.int64)
        D = np.asarray(D, dtype=np.float32)
        nf = X.shape[1]
        W = np.zeros((nf, n_actions), np.float32)
        for _ in range(iters):
            Q2 = X2 @ W
            target = R + gamma * (1.0 - D) * Q2.max(axis=1)
            W = np.zeros((nf, n_actions), np.float32)
            for a in range(n_actions):
                m = np.equal(A, a)
                if not m.any():
                    continue
                Xa, ya = X[m], target[m]
                G = Xa.T @ Xa + ridge * len(Xa) * np.eye(nf, dtype=np.float32)
                W[:, a] = np.linalg.solve(G, Xa.T @ ya).astype(np.float32)
        return LinearQ(W)

    def q(self, s: F32) -> F32:
        x = np.concatenate([np.asarray(s, np.float32), np.ones(1, np.float32)])
        return (x @ self.W).astype(np.float32)

    def act(self, s: F32) -> int:
        return int(np.argmax(self.q(s)))

    def n_params(self) -> int:
        return int(self.W.size)

    def nbytes(self) -> int:
        return int(self.W.size) * 4
