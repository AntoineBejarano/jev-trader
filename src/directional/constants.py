from __future__ import annotations

FEATURE_SCHEMA_VERSION = "directional.state.v1"
QUESTION_VERSION = "q.directional.v1"
STRATEGY_VERSION = "directional.v1"
STRATEGY_LABEL = "DIRECTIONAL_V1"

MAGNITUDE_BUCKETS = ("NOISE", "NORMAL", "LARGE", "EXTREME")
HORIZON_LABELS = ("1m", "5m", "10m", "30m")
HORIZON_SECONDS = (60, 300, 600, 1800)
CANDLE_TFS = ("1m", "5m", "10m", "30m")
