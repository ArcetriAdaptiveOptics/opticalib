"""
Thread-safe ring buffer of camera frames
========================================

Shared between two kinds of threads:

- the **producer**: the SDK (Vimba) callback thread, which calls ``push`` once
  per frame the camera delivers;
- the **consumers**: any user thread (usually the main/terminal thread) which
  calls ``wait_new`` to get frames pushed *after* the call (a frame whose
  exposure had already started may still count as "new").

"""

import threading as _th
from collections import deque as _deque


class FrameBuffer:
    """
    Fixed-size FIFO of the latest frames, with a "wait for new frames" call.

    Parameters
    ----------
    capacity : int
        Maximum number of frames kept in memory. When full, the oldest frame is
        silently dropped: the producer must never block, otherwise the camera
        would run out of buffers and stop streaming.
    """

    def __init__(self, capacity: int = 100):
        # deque(maxlen=N) is the ring buffer: append() on a full deque drops
        # the element at the opposite end (the oldest frame).
        self._frames = _deque(maxlen=int(capacity))
        # Monotonic counter of every frame ever pushed. Unlike len(_frames) it
        # never goes back (not even on clear), so consumers can use it as a
        # "clock": "wait until counter >= now + n" means "wait for n new frames".
        self._count = 0
        # A Condition is a Lock plus a wait/notify mechanism:
        # - `with self._cond:` acquires the lock (mutual exclusion);
        # - `wait_for(pred)` releases the lock while sleeping, so the producer
        #   can push, and re-acquires it when woken up by `notify_all()`.
        self._cond = _th.Condition()
        # Bumped by `interrupt()`: a waiter that sees it change knows the
        # stream was stopped and gives up instead of waiting for its timeout.
        self._generation = 0

    @property
    def capacity(self) -> int:
        """Maximum number of frames kept in memory."""
        return self._frames.maxlen

    @property
    def count(self) -> int:
        """Total number of frames pushed since creation."""
        with self._cond:
            return self._count

    def push(self, frame) -> None:
        """Add a frame (producer side: called from the SDK thread)."""
        with self._cond:
            self._frames.append(frame)
            self._count += 1
            # wake up every consumer sleeping in `wait_new`: each one
            # re-checks its own predicate and goes back to sleep if not ready
            self._cond.notify_all()

    def clear(self) -> None:
        """Drop the stored frames (the counter is not reset)."""
        with self._cond:
            self._frames.clear()

    def interrupt(self) -> None:
        """Wake up every waiter and make it raise (e.g. the stream stopped)."""
        with self._cond:
            self._generation += 1
            self._cond.notify_all()

    def snapshot(self) -> list:
        """Return a copy of the list of all stored frames, oldest first."""
        with self._cond:
            return list(self._frames)

    def wait_new(self, n: int, timeout: float | None = None) -> list:
        """
        Block until ``n`` frames newer than this call are available.

        Parameters
        ----------
        n : int
            Number of fresh frames to return.
        timeout : float, optional
            Maximum wait time in seconds. None waits forever.

        Returns
        -------
        list
            The ``n`` most recent frames, oldest first. All of them were pushed
            after the call started.

        Raises
        ------
        ValueError
            If ``n`` exceeds the buffer capacity.
        TimeoutError
            If the frames did not arrive in time.
        InterruptedError
            If ``interrupt`` was called while waiting.
        """
        if n <= 0:
            return []  # (careful: list[-0:] would be the *whole* list)
        if n > self.capacity:
            raise ValueError(
                f"Requested {n} frames, but the buffer only holds {self.capacity}."
            )
        with self._cond:
            target = self._count + n  # the "clock" value we are waiting for
            generation = self._generation
            # wait_for() loops for us: sleep -> woken by notify -> check the
            # predicate -> sleep again ... until True or until timeout.
            # len() guards against a `clear()` done meanwhile.
            ready = self._cond.wait_for(
                lambda: (self._count >= target and len(self._frames) >= n)
                or self._generation != generation,
                timeout=timeout,
            )
            if self._generation != generation:
                raise InterruptedError("Stream stopped while waiting for frames.")
            if not ready:
                raise TimeoutError(f"Timed out waiting for {n} new frames.")
            # still holding the lock: the producer cannot modify the deque
            # while we copy out the last n frames
            return list(self._frames)[-n:]
