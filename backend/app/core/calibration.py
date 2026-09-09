"""Turning a model's probabilities into probabilities that mean what they say.

A simulator emits a number it calls a probability, but nothing forces that
number to be one. If the simulated distribution is narrower than reality --
and a distribution assembled from estimated shares, estimated efficiency and
estimated pace almost always is -- then every probability is pushed away from
50% and the model reports confidence it has not earned.

The fix is a monotone map fitted on outcomes: Platt scaling, a logistic
regression on the log-odds of the raw probability. Two parameters, so it can
correct both the *scale* of the model's confidence and a systematic lean
toward one side, and monotone, so it never reorders two bets.

It is a layer over a model, not a repair of one. A slope well below 1.0 says
the underlying distributions are too narrow, and that is worth fixing at the
source. What this guarantees meanwhile is that a stated 60% is a real 60%,
which is the difference between a small edge and an imaginary one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


@dataclass(frozen=True)
class ProbabilityCalibrator:
    """``p_calibrated = sigmoid(a * logit(p_raw) + b)``.

    ``a == 1, b == 0`` is the identity, which is what an already-calibrated
    model fits to. ``a < 1`` shrinks probabilities toward 50%: the correction
    for overconfidence.
    """

    a: float = 1.0
    b: float = 0.0
    n_samples: int = 0

    def apply(self, prob: float | np.ndarray) -> float | np.ndarray:
        out = _sigmoid(self.a * _logit(prob) + self.b)
        return float(out) if np.isscalar(prob) or np.ndim(prob) == 0 else out

    @property
    def is_identity(self) -> bool:
        return abs(self.a - 1.0) < 1e-9 and abs(self.b) < 1e-9

    @property
    def confidence_retained(self) -> float:
        """Share of the model's stated deviation from 50% that survives.

        Reported because it is the number a human can argue with: 0.30 means
        the model's "65%" is really a shade over 54%.
        """
        raw = 0.65
        return float((self.apply(raw) - 0.5) / (raw - 0.5))

    @classmethod
    def identity(cls) -> "ProbabilityCalibrator":
        return cls()

    @classmethod
    def fit(cls, probs, outcomes, min_samples: int = 50
            ) -> "ProbabilityCalibrator":
        """Maximum-likelihood Platt scaling.

        Refuses to fit on a thin sample and returns the identity instead: a
        calibration map fitted on forty bets is itself an overconfident
        estimate, and applying it would trade one unmeasured error for
        another.
        """
        x = _logit(np.asarray(probs, dtype=float))
        y = np.asarray(outcomes, dtype=float)
        if x.size != y.size:
            raise ValueError("probs and outcomes must be the same length")
        if x.size < min_samples or len(np.unique(y)) < 2:
            return cls(n_samples=int(x.size))

        from scipy.optimize import minimize

        def negative_log_likelihood(theta):
            a, b = theta
            z = a * x + b
            # logaddexp form is stable where a plain log(sigmoid) underflows.
            return float(np.mean(np.logaddexp(0.0, z) - y * z))

        result = minimize(negative_log_likelihood, x0=np.array([1.0, 0.0]),
                          method="BFGS")
        if not np.all(np.isfinite(result.x)):
            return cls(n_samples=int(x.size))
        return cls(a=float(result.x[0]), b=float(result.x[1]),
                   n_samples=int(x.size))


def brier(probs, outcomes) -> float:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    return float(np.mean((p - y) ** 2))


def logloss(probs, outcomes) -> float:
    p = np.clip(np.asarray(probs, dtype=float), EPS, 1.0 - EPS)
    y = np.asarray(outcomes, dtype=float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
