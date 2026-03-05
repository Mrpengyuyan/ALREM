import logging
import time

from typing import Optional

try:
    from transformers import TrainerCallback
except Exception:
    class TrainerCallback:  # type: ignore[no-redef]
        pass


class StepTimeLoggingCallback(TrainerCallback):
    """Log elapsed training time every N global steps."""

    def __init__(self, interval_steps: int, logger: logging.Logger) -> None:
        self.interval_steps = max(1, int(interval_steps))
        self.logger = logger
        self._train_start_monotonic: Optional[float] = None

    @staticmethod
    def _format_elapsed(elapsed_sec: float) -> str:
        total_sec = max(0, int(elapsed_sec))
        hours = total_sec // 3600
        minutes = (total_sec % 3600) // 60
        seconds = total_sec % 60
        return f"{hours}:{minutes:02d}:{seconds:02d}"

    def on_train_begin(self, args, state, control, **kwargs):
        self._train_start_monotonic = time.monotonic()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if not bool(getattr(state, "is_local_process_zero", True)):
            return control
        step = int(getattr(state, "global_step", 0) or 0)
        if step > 0 and step % self.interval_steps == 0:
            elapsed_sec = 0.0
            if self._train_start_monotonic is not None:
                elapsed_sec = time.monotonic() - self._train_start_monotonic
            elapsed_text = self._format_elapsed(elapsed_sec)
            total_steps = int(getattr(state, "max_steps", 0) or 0)
            total_steps_text = str(total_steps) if total_steps > 0 else "?"
            rank = int(getattr(args, "process_index", 0) or 0)
            self.logger.info(
                "[%s] Rank:%d; Step: %d/%s;",
                elapsed_text,
                rank,
                step,
                total_steps_text,
            )
        return control
