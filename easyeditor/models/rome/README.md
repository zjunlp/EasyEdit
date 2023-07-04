# ROME
This package provides a self-contained implementation of Rank-One Model Editing (ROME).

Recall that ROME's update consists of: $u$ selection, $v_*$ optimization, and $v$ insertion.
* [`compute_u.py`](compute_u.py): Chooses a $u$ vector.
* [`compute_v.py`](compute_v.py): Choose a $v_*$ via optimization, then computes $v$.
* [`rome_main.py`](rome_main.py): Instruments main logic.
* [`rome_params.py`](rome_hparams.py): Interface for specifying hyperparameters. Inherits from the base [`params.py`](../util/hparams.py) module.

For estimating second moment statistics of keys ($C = KK$), we provide the `layer_stats` module. See the [main README](../README.md) for usage instructions.
* [`layer_stats.py`](layer_stats.py): Logic for retrieving and caching key statistics.
* [`tok_dataset.py`](tok_dataset.py): Utilities for creating a dataset of tokens.