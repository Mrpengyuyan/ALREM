import logging
from types import SimpleNamespace

import pytest

from src.step_time_logging import StepTimeLoggingCallback


def test_step_time_callback_logs_elapsed_at_interval(monkeypatch, caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("alrem.test.step_time_callback")
    logger.handlers = []
    logger.propagate = True

    callback = StepTimeLoggingCallback(interval_steps=10, logger=logger)
    args = SimpleNamespace(process_index=0)
    state = SimpleNamespace(global_step=0, max_steps=8000, is_local_process_zero=True)
    control = SimpleNamespace()

    # on_train_begin -> 100s, on_step_end(10) -> 170s
    monotonic_values = iter([100.0, 170.0])
    monkeypatch.setattr("src.step_time_logging.time.monotonic", lambda: next(monotonic_values))

    with caplog.at_level(logging.INFO):
        callback.on_train_begin(args, state, control)

        state.global_step = 9
        callback.on_step_end(args, state, control)

        state.global_step = 10
        callback.on_step_end(args, state, control)

    combined = "\n".join(record.getMessage() for record in caplog.records)
    assert "Step: 9/8000" not in combined
    assert "[0:01:10] Rank:0; Step: 10/8000;" in combined


def test_step_time_callback_uses_unknown_total_steps_when_missing(monkeypatch, caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("alrem.test.step_time_callback.unknown_total")
    logger.handlers = []
    logger.propagate = True

    callback = StepTimeLoggingCallback(interval_steps=5, logger=logger)
    args = SimpleNamespace(process_index=1)
    state = SimpleNamespace(global_step=5, max_steps=0, is_local_process_zero=True)
    control = SimpleNamespace()

    monotonic_values = iter([50.0, 65.0])
    monkeypatch.setattr("src.step_time_logging.time.monotonic", lambda: next(monotonic_values))

    with caplog.at_level(logging.INFO):
        callback.on_train_begin(args, state, control)
        callback.on_step_end(args, state, control)

    combined = "\n".join(record.getMessage() for record in caplog.records)
    assert "[0:00:15] Rank:1; Step: 5/?;" in combined
