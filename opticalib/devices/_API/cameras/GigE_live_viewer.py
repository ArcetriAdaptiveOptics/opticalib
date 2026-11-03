"""
Live viewer for GigE cameras, running in a separate process
===========================================================

Why a *process* and not a thread?

- GUI toolkits (Qt, Tk) must run their event loop in the **main thread** of a
  process, so a matplotlib window cannot live in a background thread.
- In the parent (terminal) process, the window would only refresh while the
  prompt is idle: any long command would freeze it.
- A child process has its own main thread and its own GIL: it can redraw
  forever, whatever the terminal is doing.

How do frames travel between processes?

Through a block of **shared memory**: both processes map the same RAM, so
the parent writes a frame and the child reads it, with no copy through a pipe.
Layout of the block::

    [ seq (int64) | stop (int64) | image bytes ... ]

``seq`` implements a *seqlock*, a lock-free single-writer protocol:

- the writer increments ``seq`` (now odd = "writing"), copies the image, then
  increments it again (even = "done");
- the reader copies the image only if ``seq`` is even, and keeps the copy only
  if ``seq`` did not change during the copy (otherwise the image may be half
  old and half new: a "torn" frame, which is discarded).

The worst case is a skipped frame on screen. The writer never waits for the
reader, so a slow GUI can never slow down the acquisition. (Caveat: numpy
issues no memory fences, so this relies on the CPU not reordering the stores,
as on x86; on ARM a torn frame could rarely slip through: a cosmetic glitch.)

This file has two roles:

- imported by ``opticalib`` (parent side): ``ViewerProcess`` starts the child
  and writes frames;
- run as a script (child side): ``python GigE_live_viewer.py '<json config>'``.
  It is launched by path, not as ``-m opticalib...``, so the child imports only
  numpy and matplotlib, not the whole ``opticalib`` package.
"""

import os
import sys
import json
import subprocess
import numpy as np
from multiprocessing import shared_memory

_HEADER = 2  # number of int64 slots before the image: [seq, stop]


class SharedFrame:
    """One image slot in shared memory, protected by a seqlock."""

    def __init__(self, shm: shared_memory.SharedMemory, shape, dtype):
        self.shm = shm
        # numpy "views" on the shared buffer: no data is copied, reading or
        # writing these arrays reads/writes the shared RAM directly
        self._hdr = np.ndarray((_HEADER,), dtype=np.int64, buffer=shm.buf)
        self._img = np.ndarray(shape, dtype=dtype, buffer=shm.buf, offset=_HEADER * 8)

    @classmethod
    def create(cls, shape, dtype) -> "SharedFrame":
        """Allocate a new block (parent side). The parent owns it."""
        size = _HEADER * 8 + int(np.prod(shape)) * np.dtype(dtype).itemsize
        frame = cls(shared_memory.SharedMemory(create=True, size=size), shape, dtype)
        frame._hdr[:] = 0
        return frame

    @classmethod
    def attach(cls, name: str, shape, dtype) -> "SharedFrame":
        """Map an existing block by name (child side)."""
        if sys.version_info >= (3, 13):
            shm = shared_memory.SharedMemory(name=name, track=False)
        else:
            # Before 3.13, attaching registers the block with this process'
            # resource tracker, which would *delete* it when the child exits,
            # although the parent owns it. Unregister to prevent that.
            from multiprocessing import resource_tracker

            shm = shared_memory.SharedMemory(name=name)
            resource_tracker.unregister(shm._name, "shared_memory")
        return cls(shm, shape, dtype)

    # --- writer side (parent, called from the camera SDK thread) ------------
    def write(self, img: np.ndarray) -> None:
        self._hdr[0] += 1  # odd: write in progress
        self._img[...] = img
        self._hdr[0] += 1  # even: frame complete

    def request_stop(self) -> None:
        self._hdr[1] = 1

    # --- reader side (child) ------------------------------------------------
    def stop_requested(self) -> bool:
        return bool(self._hdr[1])

    def read(self, last_seq: int):
        """
        Return ``(image, seq)`` if a new complete frame is available, else
        ``(None, last_seq)``.
        """
        seq = int(self._hdr[0])
        if seq == last_seq or seq % 2:  # nothing new, or writer busy
            return None, last_seq
        img = self._img.copy()
        if int(self._hdr[0]) != seq:  # overwritten while copying: torn
            return None, last_seq
        return img, seq

    def close(self, unlink: bool = False) -> None:
        # numpy views hold a reference to the buffer: drop them first,
        # otherwise shm.close() raises BufferError
        del self._hdr, self._img
        self.shm.close()
        if unlink:  # only the owner (parent) removes the block from the system
            self.shm.unlink()


class ViewerProcess:
    """
    Parent-side handle of a live viewer child process.

    Parameters
    ----------
    shape, dtype :
        Shape and dtype of the frames that will be displayed.
    title : str
        Window title.
    plotkwargs : dict
        Options for the plot (``title``, ``xlabel``, ``ylabel``, ``clabel``,
        plus anything accepted by ``imshow``). Must be JSON-serializable,
        since they are sent to the child on its command line.
    interval_ms : int
        Refresh period of the viewer, in milliseconds.
    """

    def __init__(self, shape, dtype, title: str, plotkwargs: dict, interval_ms: int):
        config = {
            "shape": list(shape),
            "dtype": np.dtype(dtype).str,
            "title": title,
            "interval": int(interval_ms),
            "ppid": os.getpid(),  # lets the child notice if we die
            "plot": plotkwargs,
        }
        # serialize *before* allocating anything: a non-serializable kwarg
        # raises TypeError here, with nothing to clean up
        json.dumps(plotkwargs)
        self.frame = SharedFrame.create(shape, dtype)
        config["shm"] = self.frame.shm.name
        # A Jupyter/IPython kernel may set MPLBACKEND to an inline (no window)
        # "module://..." backend: drop it so the child picks a GUI backend.
        env = dict(os.environ)
        if env.get("MPLBACKEND", "").startswith("module://"):
            del env["MPLBACKEND"]
        try:
            # start_new_session: the child gets its own process group, so a
            # Ctrl-C in the terminal (SIGINT to the foreground group) does not
            # kill the viewer too. POSIX only, ignored on Windows.
            self.proc = subprocess.Popen(
                [sys.executable, __file__, json.dumps(config)],
                env=env,
                start_new_session=True,
            )
        except Exception:
            self.frame.close(unlink=True)  # do not leak the shared memory
            raise
        self.owns_stream = False  # set by the camera: stop streaming on close?

    def write(self, img: np.ndarray) -> None:
        self.frame.write(img)

    def wait(self) -> int:
        """Block until the child exits (e.g. the window is closed)."""
        return self.proc.wait()

    def close(self, timeout: float = 2.0) -> None:
        """Ask the child to quit (kill it if it does not), free the memory."""
        self.frame.request_stop()
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        self.frame.close(unlink=True)


def _run_viewer(config: dict) -> None:
    """Child process main: show the shared frame until closed or stopped."""
    import matplotlib.pyplot as plt

    shape, dtype = tuple(config["shape"]), np.dtype(config["dtype"])
    frame = SharedFrame.attach(config["shm"], shape, dtype)

    plot = config["plot"]
    title = plot.pop("title", config["title"])
    xlabel, ylabel = plot.pop("xlabel", "px"), plot.pop("ylabel", "px")
    clabel = plot.pop("clabel", "adu")
    autoscale = "vmin" not in plot and "vmax" not in plot

    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title(config["title"])
    im = ax.imshow(np.zeros(shape, dtype=dtype), **plot)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    fig.colorbar(im, ax=ax, aspect=10).set_label(clabel)

    last_seq = 0

    def refresh():
        # Runs in *this* process' main thread, inside the GUI event loop,
        # every `interval` ms. It must be quick: while it runs, the window
        # cannot respond to the mouse.
        nonlocal last_seq
        if frame.stop_requested() or os.getppid() != config["ppid"]:
            plt.close(fig)  # makes plt.show() below return
            return
        img, last_seq = frame.read(last_seq)
        if img is not None:
            im.set_data(img)
            if autoscale:
                im.autoscale()
            fig.canvas.draw_idle()  # schedule a redraw, do not draw right now

    timer = fig.canvas.new_timer(interval=config["interval"])
    timer.add_callback(refresh)
    timer.start()
    plt.show()  # blocking: this process' main thread now runs the event loop
    timer.stop()
    frame.close()


if __name__ == "__main__":
    _run_viewer(json.loads(sys.argv[1]))
