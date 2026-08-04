"""
LSF: Layer Selection + Fusion (ICLR 2027 revision of MM++).

Public surface:

    from lsf import (
        LayerBlocks, JointCovariance, build_joint_covariance,
        novelty_logdet, canonical_correlations, select_layers,
    )

Design note: everything here reads from ONE jointly-shrunk covariance over all
candidate layers (joint_stats.py). That is what keeps every conditional
covariance PSD and lets the selector, the diagnostics and the oracle sweep share
a single pass over the features.
"""

from .cache_io import (
    CacheLayout,
    auroc,
    check_shard_order,
    fpr_at_tpr,
    load_class_means,
    load_layer_features,
    load_train_labels,
    preflight,
)
from .joint_stats import (
    JointCovariance,
    StreamingJointCovariance,
    build_joint_covariance,
    joint_covariance_from_features,
    ledoit_wolf_from_accumulators,
    shrink,
)
from .novelty import (
    LayerBlocks,
    canonical_correlations,
    conditional_covariance,
    novelty_logdet,
    novelty_scores,
    novelty_trace,
)
from .scoring import (
    block_diagonal_precision,
    fuse_features,
    fused_class_means,
    maha_scores,
    per_layer_maha_scores,
    score_subset,
)
from .selector import (
    SelectionResult,
    entropy_density,
    entropy_density_drops,
    select_layers,
)

__all__ = [
    'JointCovariance', 'StreamingJointCovariance', 'build_joint_covariance',
    'joint_covariance_from_features', 'ledoit_wolf_from_accumulators', 'shrink',
    'LayerBlocks', 'canonical_correlations', 'conditional_covariance',
    'novelty_logdet', 'novelty_scores', 'novelty_trace',
    'SelectionResult', 'entropy_density', 'entropy_density_drops', 'select_layers',
    'CacheLayout', 'preflight', 'check_shard_order', 'load_train_labels',
    'load_layer_features', 'load_class_means', 'auroc', 'fpr_at_tpr',
    'score_subset', 'maha_scores', 'per_layer_maha_scores', 'fuse_features',
    'fused_class_means', 'block_diagonal_precision',
]
