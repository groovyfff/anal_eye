"""Feature engineering subpackage.

``schema`` defines the canonical ~60-feature contract shared between training
and live inference. ``engineering`` computes those features from raw OHLCV +
microstructure data (open interest, CVD, order-flow imbalance) using TA-Lib.
"""

from ae_brain.features.schema import FEATURE_NAMES, FEATURE_SCHEMA, n_features
from ae_brain.features.engineering import FeatureEngineer

__all__ = ["FEATURE_NAMES", "FEATURE_SCHEMA", "n_features", "FeatureEngineer"]
