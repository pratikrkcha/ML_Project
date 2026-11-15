"""Attack simulation module for FedDef (Task 3).

Implements model-poisoning attack functions that corrupt client parameter
updates before they are submitted to the aggregation server.

Supported attack types (matching config.py ALLOWED_ATTACK_TYPES):
- random     : Replace client gradients/params with scaled random noise.
                Directly mirrors the prototype in maskeensingh/FEDDEF.
- sign_flip  : Negate all gradient components (Byzantine sign-flip).
- scaling    : Amplify the update magnitude to bypass anomaly detection.
- label_flip : Equivalent to training on negated labels (simulated here as
               negated coef + mirrored intercept).
- collusion  : Identical to random for a single-round attack; in a multi-
               round colluding scenario all malicious clients submit the
               same random vector, increasing its weight under plain FedAvg.

Author: Kunal V Kurundwadkar (IIT2024177) — Task 3
Date: 14 November 2026
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

Params = dict[str, Any]
AttackFn = Callable[[Params, np.random.Generator], Params]


# ---------------------------------------------------------------------------
# Individual attack implementations
# ---------------------------------------------------------------------------


def random_noise_attack(
    params: Params,
    rng: np.random.Generator,
    *,
    strength: float = 1.0,
) -> Params:
    """Replace client parameters with scaled random noise.

    The standard deviation is `strength` per coefficient and ten times
    `strength` for the scalar intercept, matching the Task 3 simulation.
    """
    coef = np.asarray(params["coef"], dtype=float).ravel()
    noisy_coef = rng.normal(0.0, strength, size=coef.shape)
    noisy_intercept = float(rng.normal(0.0, 10.0 * strength))
    return {"coef": noisy_coef, "intercept": noisy_intercept}


def sign_flip_attack(
    params: Params,
    rng: np.random.Generator,
    *,
    strength: float = 1.0,
) -> Params:
    """Negate all parameter components (Byzantine sign-flip attack).

    The malicious client sends the negative of its true update, pushing the
    global model in the wrong direction.
    """
    coef = np.asarray(params["coef"], dtype=float).ravel()
    intercept = float(params["intercept"])
    return {
        "coef": -coef * strength,
        "intercept": -intercept * strength,
    }


def scaling_attack(
    params: Params,
    rng: np.random.Generator,
    *,
    strength: float = 10.0,
) -> Params:
    """Amplify the update magnitude (scaling / boosting attack).

    By returning a large-magnitude update, the malicious client tries to
    dominate the weighted average and bypass simple magnitude-based detection.
    """
    coef = np.asarray(params["coef"], dtype=float).ravel()
    intercept = float(params["intercept"])
    scale = max(strength, 1.0)
    return {
        "coef": coef * scale,
        "intercept": intercept * scale,
    }


def label_flip_attack(
    params: Params,
    rng: np.random.Generator,
    *,
    strength: float = 1.0,
) -> Params:
    """Simulate label-flip attack by negating the coefficient vector.

    A true label-flip attack trains the local model on swapped labels
    (0→1, 1→0).  For a linear model this approximately negates the optimal
    weight vector, so we negate coef and mirror intercept.
    """
    coef = np.asarray(params["coef"], dtype=float).ravel()
    intercept = float(params["intercept"])
    return {
        "coef": -coef * strength,
        "intercept": -intercept + strength * 0.1,
    }


def collusion_attack(
    params: Params,
    rng: np.random.Generator,
    *,
    strength: float = 1.0,
    shared_seed: int = 0,
) -> Params:
    """Colluding clients all send the same random poisoning vector.

    In a true colluding scenario every malicious client uses the same RNG
    seed so they submit identical updates; this amplifies their effect under
    un-defended FedAvg proportional to their count.  Under FEDDEF, their
    identical predictions cluster tightly, making them *easier* to detect.
    """
    shared_rng = np.random.default_rng(shared_seed)
    coef = np.asarray(params["coef"], dtype=float).ravel()
    n_features = coef.shape[0]
    noisy_coef = shared_rng.standard_normal(n_features) * strength
    noisy_intercept = float(shared_rng.standard_normal() * strength * 10.0)
    return {"coef": noisy_coef, "intercept": noisy_intercept}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_ATTACK_REGISTRY: dict[str, AttackFn] = {
    "random": random_noise_attack,
    "sign_flip": sign_flip_attack,
    "scaling": scaling_attack,
    "label_flip": label_flip_attack,
    "collusion": collusion_attack,
}


def make_attack_fn(attack_type: str, *, strength: float = 1.0) -> AttackFn:
    """Return a bound attack callable for the given type and strength.

    Parameters
    ----------
    attack_type:
        One of "random", "sign_flip", "scaling", "label_flip", "collusion".
    strength:
        Attack magnitude; meaning is attack-specific (see individual docs).

    Returns
    -------
    fn : (params, rng) -> params
        A callable that takes a client's Params dict and an RNG and returns
        a corrupted Params dict.
    """
    if attack_type not in _ATTACK_REGISTRY:
        raise ValueError(
            f"Unknown attack_type {attack_type!r}. "
            f"Choose from {sorted(_ATTACK_REGISTRY.keys())}."
        )
    base_fn = _ATTACK_REGISTRY[attack_type]

    def bound_attack(params: Params, rng: np.random.Generator) -> Params:
        return base_fn(params, rng, strength=strength)

    bound_attack.__name__ = f"{attack_type}_attack_s{strength}"
    return bound_attack


def list_attacks() -> list[str]:
    """Return all registered attack type names."""
    return sorted(_ATTACK_REGISTRY.keys())
