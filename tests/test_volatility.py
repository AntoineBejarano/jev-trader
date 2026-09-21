from __future__ import annotations

from src.features.volatility import realized_variance, sampled_log_returns


def test_constant_price_has_zero_realized_vol() -> None:
    timestamps = list(range(0, 5_000, 1_000))
    values = [100.0] * len(timestamps)
    rets = sampled_log_returns(timestamps, values, now_ms=4_000, window_ms=4_000, step_ms=1_000)
    assert all(abs(value) < 1e-12 for value in rets)
    rv = realized_variance(rets)
    assert rv is not None
    assert rv == 0.0


def test_two_point_move_is_log_return_magnitude() -> None:
    timestamps = [0, 1_000]
    values = [100.0, 101.0]
    rets = sampled_log_returns(timestamps, values, now_ms=1_000, window_ms=1_000, step_ms=1_000)
    assert len(rets) == 1
    rv = realized_variance(rets)
    assert rv is not None
    assert abs(rv - abs(rets[0])) < 1e-12
