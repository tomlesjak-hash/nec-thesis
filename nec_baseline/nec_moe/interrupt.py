"""Clean interruption (brief 07 C.3).

On the **first** SIGINT (Ctrl+C) or SIGTERM a run is not killed: a stop is
*requested*. The trainer notices it after the current optimizer step, writes
a checkpoint and raises :class:`RunInterrupted`; the run store then records
the run as ``interrupted``, and it can be resumed exactly. A **second** signal
exits immediately (the default behaviour, restored by the first).

Outside :func:`graceful_interrupts` nothing changes: signals behave as usual,
and :func:`stop_requested` is false unless code requests a stop itself (which
is how the tests simulate an interruption at a chosen step).
"""

from __future__ import annotations

import contextlib
import signal
import sys
import threading
from collections.abc import Iterator
from types import FrameType

__all__ = [
    "RunInterrupted",
    "request_stop",
    "stop_requested",
    "clear_stop",
    "graceful_interrupts",
]

_STOP = threading.Event()


class RunInterrupted(KeyboardInterrupt):
    """A requested stop: the training state was checkpointed before raising."""


def request_stop() -> None:
    _STOP.set()


def stop_requested() -> bool:
    return _STOP.is_set()


def clear_stop() -> None:
    _STOP.clear()


@contextlib.contextmanager
def graceful_interrupts() -> Iterator[None]:
    """Turn the first SIGINT/SIGTERM into a stop request; the second exits.

    Only installable from the main thread (a Python rule); elsewhere it is a
    no-op, and signals keep their default behaviour.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    signals = (signal.SIGINT, signal.SIGTERM)
    previous = {s: signal.getsignal(s) for s in signals}

    def second(signum: int, frame: FrameType | None) -> None:
        raise KeyboardInterrupt(f"second signal {signum}: stopping immediately")

    def first(signum: int, frame: FrameType | None) -> None:
        request_stop()
        print(f"\n[interrupt] signal {signum}: finishing the current step, writing "
              "a checkpoint, then stopping. Send it again to stop immediately.",
              file=sys.stderr)
        for s in signals:
            signal.signal(s, second)

    clear_stop()
    for s in signals:
        signal.signal(s, first)
    try:
        yield
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)
        clear_stop()
