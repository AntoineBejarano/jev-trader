from __future__ import annotations

import pytest
from src.market.orderbook import LocalOrderBook, book_imbalance


def test_book_imbalance_top_of_book() -> None:
    assert book_imbalance(10, 5) == pytest.approx(1 / 3)
    assert book_imbalance(0, 0) is None


def test_partial_snapshot_and_level_imbalance() -> None:
    book = LocalOrderBook()
    book.apply_partial_snapshot(
        bids=[(100.0, 10.0), (99.0, 10.0), (98.0, 10.0), (97.0, 1.0), (96.0, 1.0)],
        asks=[(101.0, 5.0), (102.0, 5.0), (103.0, 5.0), (104.0, 1.0), (105.0, 1.0)],
        last_update_id=1,
        event_time_ms=1_000,
    )
    assert book.valid
    assert book.mid == 100.5
    assert book.spread == 1.0
    assert book.imbalance(1) == pytest.approx((10 - 5) / 15)
    assert book.imbalance(5) == pytest.approx((32 - 17) / 49)


def test_crossed_book_is_invalid() -> None:
    book = LocalOrderBook()
    book.apply_partial_snapshot(
        bids=[(102.0, 1.0)],
        asks=[(101.0, 1.0)],
        last_update_id=2,
        event_time_ms=1_000,
    )
    assert book.valid is False


def test_book_ticker_updates_best_quotes() -> None:
    book = LocalOrderBook()
    book.apply_partial_snapshot(
        bids=[(100.0, 1.0), (99.0, 2.0)],
        asks=[(101.0, 1.0), (102.0, 2.0)],
        last_update_id=3,
        event_time_ms=1_000,
    )
    book.apply_book_ticker(100.2, 4.0, 100.4, 1.0, update_id=4, event_time_ms=1_100)
    assert book.best_bid == 100.2
    assert book.best_ask == 100.4
    assert book.imbalance(1) == pytest.approx((4 - 1) / 5)


def test_clear_invalidates_book() -> None:
    book = LocalOrderBook()
    book.apply_partial_snapshot(
        bids=[(100.0, 1.0)],
        asks=[(101.0, 1.0)],
        last_update_id=1,
        event_time_ms=1,
    )
    book.clear()
    assert book.valid is False
    assert book.mid is None
