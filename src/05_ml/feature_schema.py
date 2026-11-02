"""Shared reaction-feature names and encoding for training, inference and plots."""

import numpy as np

REACTION_PREFIXES = ("rxn_", "flux_")
PRESENCE_THRESHOLD = 1e-9


def is_reaction_feature(name):
    return str(name).startswith(REACTION_PREFIXES)


def reaction_id(name):
    """Remove one feature prefix, including the prefix used by the trained models."""
    for prefix in REACTION_PREFIXES:
        if str(name).startswith(prefix):
            return str(name)[len(prefix):]
    raise ValueError(f"Not a reaction feature: {name!r}")


def reaction_activity(values):
    """Use the same binary activity representation in training and prediction."""
    return (np.abs(np.asarray(values)) > PRESENCE_THRESHOLD).astype(int)


def reaction_feature_row(flux_row, feature_names, reaction_indices):
    """Build a prediction row without silently replacing unknown reactions by zero."""
    result = {}
    for name in feature_names:
        rid = reaction_id(name)
        if rid not in reaction_indices:
            raise ValueError(f"Reaction {rid!r} required by {name!r} is missing")
        result[name] = int(reaction_activity(flux_row[reaction_indices[rid]]))
    return result
