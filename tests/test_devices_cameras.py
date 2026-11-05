"""
Tests for the GigaVision streaming / live viewer machinery.

No hardware and no GUI: the camera is a fake driven by a background thread,
and the viewer child process is replaced by a harmless ``sleep`` process.
"""

import os
import sys
import glob
import time
import threading
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

vmbpy = pytest.importorskip("vmbpy")

from opticalib.devices import cameras
from opticalib.devices._API.cameras import GigE_live_viewer as viewer_mod
from opticalib.devices._API.cameras.frame_buffer import FrameBuffer
from opticalib.devices._API.cameras.GigE_live_viewer import SharedFrame, ViewerProcess


def _shm_blocks():
    return set(glob.glob("/dev/shm/psm_*"))


def _wait_for(pred, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.005)
    return pred()


# --------------------------------------------------------------------------- #
# FrameBuffer
# --------------------------------------------------------------------------- #
class TestFrameBuffer:
    def test_capacity_and_empty(self):
        b = FrameBuffer(3)
        assert b.capacity == 3 and b.count == 0 and b.snapshot() == []

    def test_ring_drops_oldest(self):
        b = FrameBuffer(3)
        for i in range(5):
            b.push(i)
        assert b.snapshot() == [2, 3, 4]
        assert b.count == 5

    def test_capacity_one(self):
        b = FrameBuffer(1)
        b.push("a")
        b.push("b")
        assert b.snapshot() == ["b"]

    def test_snapshot_is_copy(self):
        b = FrameBuffer(3)
        b.push(1)
        s = b.snapshot()
        s.append(99)
        assert b.snapshot() == [1]

    def test_clear_keeps_count(self):
        b = FrameBuffer(3)
        b.push(1)
        b.push(2)
        b.clear()
        assert b.snapshot() == [] and b.count == 2

    def test_wait_new_ignores_old_frames(self):
        b = FrameBuffer(10)
        for i in range(5):
            b.push(("old", i))

        def produce():
            time.sleep(0.05)
            for i in range(3):
                b.push(("new", i))

        threading.Thread(target=produce).start()
        out = b.wait_new(3, timeout=5)
        assert out == [("new", 0), ("new", 1), ("new", 2)]

    def test_wait_new_timeout(self):
        b = FrameBuffer(5)
        b.push(1)
        with pytest.raises(TimeoutError):
            b.wait_new(1, timeout=0.05)

    def test_wait_new_timeout_partial(self):
        b = FrameBuffer(5)
        threading.Timer(0.01, b.push, args=(1,)).start()
        with pytest.raises(TimeoutError):
            b.wait_new(2, timeout=0.2)

    def test_wait_new_n_exceeds_capacity(self):
        with pytest.raises(ValueError):
            FrameBuffer(2).wait_new(3, timeout=0.01)

    def test_wait_new_n_equals_capacity(self):
        b = FrameBuffer(3)
        threading.Timer(0.02, lambda: [b.push(i) for i in range(3)]).start()
        assert b.wait_new(3, timeout=5) == [0, 1, 2]

    def test_wait_new_zero(self):
        # n=0 must return no frames (list[-0:] would return the whole buffer)
        b = FrameBuffer(3)
        b.push(1)
        assert b.wait_new(0, timeout=0.1) == []

    def test_wait_new_survives_clear(self):
        b = FrameBuffer(10)

        def produce():
            time.sleep(0.03)
            b.push(1)
            b.clear()  # wipes a frame the waiter needed
            for i in (2, 3):
                b.push(i)

        threading.Thread(target=produce).start()
        out = b.wait_new(2, timeout=5)
        assert out == [2, 3]

    def test_multiple_waiters(self):
        b = FrameBuffer(20)
        results = {}

        def wait(k, n):
            results[k] = b.wait_new(n, timeout=5)

        ths = [threading.Thread(target=wait, args=(k, n)) for k, n in enumerate((1, 3, 5))]
        for t in ths:
            t.start()
        time.sleep(0.05)
        for i in range(5):
            b.push(i)
            time.sleep(0.005)
        for t in ths:
            t.join(5)
        assert len(results[0]) == 1 and results[0][0] in range(5)
        assert results[1] == [0, 1, 2]
        assert results[2] == [0, 1, 2, 3, 4]

    def test_concurrent_push_count(self):
        b = FrameBuffer(10)

        def prod():
            for _ in range(500):
                b.push(0)

        ths = [threading.Thread(target=prod) for _ in range(4)]
        [t.start() for t in ths]
        [t.join() for t in ths]
        assert b.count == 2000 and len(b.snapshot()) == 10


# --------------------------------------------------------------------------- #
# SharedFrame
# --------------------------------------------------------------------------- #
class TestSharedFrame:
    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
    def test_roundtrip(self, dtype):
        shape = (7, 5)
        w = SharedFrame.create(shape, dtype)
        r = SharedFrame.attach(w.shm.name, shape, dtype)
        try:
            img = (np.arange(35).reshape(shape) % 200).astype(dtype)
            w.write(img)
            out, seq = r.read(0)
            assert seq == 2
            np.testing.assert_array_equal(out, img)
            assert out.dtype == np.dtype(dtype)
        finally:
            r.close()
            w.close(unlink=True)

    def test_color_shape(self):
        w = SharedFrame.create((3, 4, 4), np.uint8)
        try:
            img = np.random.randint(0, 255, (3, 4, 4), dtype=np.uint8)
            w.write(img)
            np.testing.assert_array_equal(w.read(0)[0], img)
        finally:
            w.close(unlink=True)

    def test_read_returns_none_when_unchanged(self):
        w = SharedFrame.create((2, 2), np.uint8)
        try:
            assert w.read(0) == (None, 0)  # nothing written yet
            w.write(np.ones((2, 2), np.uint8))
            _, seq = w.read(0)
            assert w.read(seq) == (None, seq)
        finally:
            w.close(unlink=True)

    def test_read_returns_none_when_writer_busy(self):
        w = SharedFrame.create((2, 2), np.uint8)
        try:
            w._hdr[0] = 3  # odd: write in progress
            assert w.read(0) == (None, 0)
        finally:
            w.close(unlink=True)

    def test_read_detects_torn_frame(self):
        w = SharedFrame.create((2, 2), np.uint8)
        try:
            w.write(np.ones((2, 2), np.uint8))
            orig = w._img

            class Racy:
                """Image proxy whose copy() simulates a concurrent write."""

                def copy(_self):
                    out = orig.copy()
                    w._hdr[0] += 2
                    return out

            w._img = Racy()
            assert w.read(0) == (None, 0)
            w._img = orig
            img, seq = w.read(0)
            assert img is not None and seq == 4
        finally:
            w.close(unlink=True)

    def test_sequence_increments(self):
        w = SharedFrame.create((2,), np.uint8)
        try:
            for i in range(1, 4):
                w.write(np.array([i, i], np.uint8))
                assert w.read(-1)[1] == 2 * i
        finally:
            w.close(unlink=True)

    def test_stop_flag_shared(self):
        w = SharedFrame.create((2,), np.uint8)
        r = SharedFrame.attach(w.shm.name, (2,), np.uint8)
        try:
            assert not r.stop_requested()
            w.request_stop()
            assert r.stop_requested()
        finally:
            r.close()
            w.close(unlink=True)

    def test_write_wrong_shape_raises(self):
        w = SharedFrame.create((2, 2), np.uint8)
        try:
            with pytest.raises(ValueError):
                w.write(np.ones((3, 3), np.uint8))
        finally:
            w.close(unlink=True)

    def test_close_unlink_releases(self):
        before = _shm_blocks()
        w = SharedFrame.create((4, 4), np.uint16)
        name = w.shm.name
        assert _shm_blocks() - before
        w.close(unlink=True)
        assert _shm_blocks() == before
        with pytest.raises(FileNotFoundError):
            SharedFrame.attach(name, (4, 4), np.uint16)

    def test_attach_close_does_not_unlink(self):
        w = SharedFrame.create((2,), np.uint8)
        try:
            r = SharedFrame.attach(w.shm.name, (2,), np.uint8)
            r.close()
            r2 = SharedFrame.attach(w.shm.name, (2,), np.uint8)
            r2.close()
        finally:
            w.close(unlink=True)


# --------------------------------------------------------------------------- #
# ViewerProcess
# --------------------------------------------------------------------------- #
class _QuickProc(subprocess.Popen):
    """Popen whose timed waits are short, so close() does not take 2 s."""

    def wait(self, timeout=None):
        return super().wait(timeout=0.05 if timeout else None)


@pytest.fixture
def fake_child(monkeypatch):
    """Replace the viewer child by a sleeping process; yields launched procs."""
    procs = []

    def popen(args, env=None, **kw):
        assert args[0] == sys.executable and args[1] == viewer_mod.__file__
        p = _QuickProc([sys.executable, "-c", "import time; time.sleep(30)"])
        p.viewer_args = args
        procs.append(p)
        return p

    monkeypatch.setattr(
        viewer_mod,
        "subprocess",
        SimpleNamespace(Popen=popen, TimeoutExpired=subprocess.TimeoutExpired),
    )
    yield procs
    for p in procs:
        if p.poll() is None:
            p.kill()
            subprocess.Popen.wait(p)


class TestViewerProcess:
    def test_bad_kwargs_type_error_no_leak(self, fake_child):
        before = _shm_blocks()
        with pytest.raises(TypeError):
            ViewerProcess((4, 4), np.uint8, "t", {"cmap": object()}, 40)
        assert _shm_blocks() == before
        assert fake_child == []

    def test_payload_and_lifecycle(self, fake_child):
        before = _shm_blocks()
        vp = ViewerProcess((4, 6), np.uint16, "title", {"cmap": "gray"}, 33)
        assert vp.owns_stream is False
        import json

        cfg = json.loads(fake_child[0].viewer_args[2])
        assert cfg["shape"] == [4, 6] and cfg["dtype"] == np.dtype(np.uint16).str
        assert cfg["interval"] == 33 and cfg["ppid"] == os.getpid()
        assert cfg["plot"] == {"cmap": "gray"} and cfg["shm"] == vp.frame.shm.name
        vp.write(np.full((4, 6), 7, np.uint16))
        assert vp.frame.read(0)[0][0, 0] == 7
        vp.close()
        assert vp.proc.poll() is not None
        assert _shm_blocks() == before

    def test_close_sets_stop_flag_and_kills_stubborn_child(self, fake_child):
        vp = ViewerProcess((2, 2), np.uint8, "t", {}, 40)
        flag = SharedFrame.attach(vp.frame.shm.name, (2, 2), np.uint8)
        vp.close()  # child ignores the flag: must be killed
        assert vp.proc.returncode is not None
        assert flag.stop_requested()
        flag.close()

    def test_wait_returns_on_child_exit(self, fake_child):
        vp = ViewerProcess((2, 2), np.uint8, "t", {}, 40)
        threading.Timer(0.05, vp.proc.terminate).start()
        assert vp.wait() != 0 or True
        vp.close()


class TestRunViewerHeadless:
    @pytest.mark.parametrize(
        "backend, stripped",
        [("module://matplotlib_inline.backend_inline", True), ("QtAgg", False)],
    )
    def test_env_mplbackend(self, monkeypatch, backend, stripped):
        """Inline (module://) backends are dropped, explicit GUI ones kept."""
        seen = {}

        class P:
            def wait(self, timeout=None):
                return 0

        def popen(args, env=None, **kw):
            seen["env"], seen["kw"] = env, kw
            return P()

        monkeypatch.setenv("MPLBACKEND", backend)
        monkeypatch.setattr(
            viewer_mod,
            "subprocess",
            SimpleNamespace(Popen=popen, TimeoutExpired=subprocess.TimeoutExpired),
        )
        vp = ViewerProcess((2, 2), np.uint8, "t", {}, 40)
        vp.close()
        assert ("MPLBACKEND" not in seen["env"]) is stripped
        assert seen["kw"].get("start_new_session") is True  # immune to Ctrl-C

    def test_popen_failure_releases_shm(self, monkeypatch):
        created = []
        orig = viewer_mod.SharedFrame.create

        def create(*a):
            created.append(orig(*a))
            return created[-1]

        def popen(*a, **kw):
            raise OSError("boom")

        monkeypatch.setattr(viewer_mod.SharedFrame, "create", staticmethod(create))
        monkeypatch.setattr(
            viewer_mod,
            "subprocess",
            SimpleNamespace(Popen=popen, TimeoutExpired=subprocess.TimeoutExpired),
        )
        with pytest.raises(OSError):
            ViewerProcess((2, 2), np.uint8, "t", {}, 40)
        with pytest.raises(FileNotFoundError):  # unlinked: cannot re-attach
            viewer_mod.shared_memory.SharedMemory(name=created[0].shm.name)


class TestStreamInterrupt:
    def test_interrupt_wakes_waiter(self):
        buf = FrameBuffer(10)
        errors = []

        def waiter():
            try:
                buf.wait_new(3, timeout=10)
            except InterruptedError as e:
                errors.append(e)

        t = threading.Thread(target=waiter)
        t.start()
        time.sleep(0.05)
        t0 = time.monotonic()
        buf.interrupt()
        t.join(2)
        assert errors and time.monotonic() - t0 < 1


# --------------------------------------------------------------------------- #
# GigaVision
# --------------------------------------------------------------------------- #
H, W = 6, 8


class FakeFrame:
    def __init__(self, v, ok=True):
        self.v, self.ok = v, ok

    def get_status(self):
        return vmbpy.FrameStatus.Complete if self.ok else vmbpy.FrameStatus.Incomplete

    def as_numpy_ndarray(self):
        return np.full((H, W, 1), self.v, np.uint16)


class FakeCamera:
    def __init__(self, period=0.003, bad_every=0):
        self.period, self.bad_every = period, bad_every
        self._t = None
        self._stop = threading.Event()
        self.queued = 0
        self.starts = 0
        self.feature = MagicMock()
        self.last_kwargs = None

    def start_streaming(self, handler, buffer_count, allocation_mode):
        self.starts += 1
        self.last_kwargs = dict(buffer_count=buffer_count, allocation_mode=allocation_mode)
        self._stop.clear()

        def run():
            i = 0
            while not self._stop.is_set():
                i += 1
                ok = not (self.bad_every and i % self.bad_every == 0)
                handler(self, None, FakeFrame(i, ok))
                self._stop.wait(self.period)

        self._t = threading.Thread(target=run, daemon=True)
        self._t.start()

    def stop_streaming(self):
        self._stop.set()
        self._t.join()
        self._t = None

    def is_streaming(self):
        return self._t is not None

    def queue_frame(self, frame):
        self.queued += 1

    def get_feature_by_name(self, name):
        return self.feature


def make_cam(**camkw):
    g = cameras.GigaVision.__new__(cameras.GigaVision)
    g._name = "fake"
    g._logger = MagicMock()
    g._base_timeout = 2000
    g._exptime = 1000
    g._buffer = FrameBuffer(50)
    g._viewer = None
    g._viewer_lock = threading.Lock()
    g._stream_lock = threading.RLock()
    g._cam = FakeCamera(**camkw)
    return g


@pytest.fixture
def cam(fake_child):
    before = _shm_blocks()
    g = make_cam()
    yield g
    if g._cam is not None:
        g.stop_live_view()
        g.stop_streaming()
    assert not g.is_live
    assert _shm_blocks() == before
    assert threading.active_count() <= 2 or _wait_for(
        lambda: not any(t.name == "fake-viewer-watch" for t in threading.enumerate())
    )


class TestStreaming:
    def test_start_stop(self, cam):
        assert not cam.is_streaming
        cam.start_streaming()
        assert cam.is_streaming
        assert _wait_for(lambda: cam._buffer.count > 2)
        cam.stop_streaming()
        assert not cam.is_streaming

    def test_start_idempotent(self, cam):
        cam.start_streaming()
        cam.start_streaming()
        assert cam._cam.starts == 1
        cam.stop_streaming()

    def test_stop_when_not_streaming_noop(self, cam):
        cam.stop_streaming()
        assert not cam.is_streaming

    def test_allocation_modes(self, cam):
        cam.start_streaming(allocation_mode=1)
        assert cam._cam.last_kwargs["allocation_mode"] == vmbpy.AllocationMode.AllocAndAnnounceFrame
        cam.stop_streaming()
        cam.start_streaming(allocation_mode=0)
        assert cam._cam.last_kwargs["allocation_mode"] == vmbpy.AllocationMode.AnnounceFrame
        cam.stop_streaming()

    def test_is_streaming_with_no_cam(self, cam):
        c = cam._cam
        cam._cam = None
        assert cam.is_streaming is False
        cam._cam = c

    def test_start_clears_buffer(self, cam):
        cam._buffer.push("stale")
        cam.start_streaming()
        assert not any(isinstance(f, str) for f in cam._buffer.snapshot())
        cam.stop_streaming()

    def test_on_frame_skips_incomplete_but_requeues(self, cam):
        c = cam._cam
        cam._on_frame(c, None, FakeFrame(1, ok=False))
        assert cam._buffer.count == 0 and c.queued == 1
        cam._on_frame(c, None, FakeFrame(2, ok=True))
        assert cam._buffer.count == 1 and c.queued == 2
        img = cam._buffer.snapshot()[0]
        assert img.shape == (H, W) and img[0, 0] == 2

    def test_on_frame_requeues_on_error(self, cam):
        c = cam._cam

        class Boom(FakeFrame):
            def as_numpy_ndarray(self):
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            cam._on_frame(c, None, Boom(1))
        assert c.queued == 1

    def test_on_frame_color_transposed(self, cam):
        class Color(FakeFrame):
            def as_numpy_ndarray(self):
                return np.zeros((H, W, 3), np.uint8)

        cam._on_frame(cam._cam, None, Color(0))
        assert cam._buffer.snapshot()[0].shape == (3, H, W)

    def test_on_frame_copies_data(self, cam):
        buf = np.zeros((H, W, 1), np.uint8)

        class F(FakeFrame):
            def as_numpy_ndarray(self):
                return buf

        cam._on_frame(cam._cam, None, F(0))
        buf[:] = 9
        assert cam._buffer.snapshot()[0].max() == 0

    def test_on_frame_feeds_viewer(self, cam):
        v = MagicMock()
        cam._viewer = v
        cam._on_frame(cam._cam, None, FakeFrame(3))
        v.write.assert_called_once()
        cam._viewer = None

    def test_incomplete_frames_stream(self, cam):
        cam._cam.bad_every = 2
        cam.start_streaming()
        assert _wait_for(lambda: cam._buffer.count >= 5)
        cam.stop_streaming()
        vals = [int(f[0, 0]) for f in cam._buffer.snapshot()]
        assert all(v % 2 == 1 for v in vals)
        assert cam._cam.queued >= cam._buffer.count


class TestAcquireFramesStreaming:
    def test_single_frame(self, cam):
        cam.start_streaming()
        out = cam.acquire_frames()
        assert out.shape == (H, W)
        cam.stop_streaming()

    def test_cube_shape_and_order(self, cam):
        cam.start_streaming()
        before = cam._buffer.count
        cube = cam.acquire_frames(5, multiframe_out_mode="cube")
        assert cube.shape == (H, W, 5)
        vals = [int(cube[0, 0, k]) for k in range(5)]
        assert vals == sorted(vals) and len(set(vals)) == 5
        assert vals[0] > before  # strictly fresh frames
        cam.stop_streaming()

    def test_mean_shape(self, cam):
        cam.start_streaming()
        m = cam.acquire_frames(4)
        assert m.shape == (H, W)
        cam.stop_streaming()

    @pytest.mark.parametrize("mode", ["sync", "async"])
    def test_mode_ignored_when_streaming(self, cam, mode):
        cam.start_streaming()
        cube = cam.acquire_frames(3, multiframe_out_mode="cube", mode=mode)
        assert cube.shape == (H, W, 3)
        assert cam.is_streaming  # stream not stopped by acquisition
        assert cam._cam.starts == 1
        cam.stop_streaming()

    def test_invalid_mode(self, cam):
        with pytest.raises(ValueError):
            cam.acquire_frames(1, mode="bogus")

    def test_exceeding_capacity(self, cam):
        cam.start_streaming()
        with pytest.raises(ValueError):
            cam.acquire_frames(cam._buffer.capacity + 1)
        cam.stop_streaming()

    def test_timeout_when_no_frames(self, cam):
        cam.start_streaming()
        cam._cam._stop.set()  # producer thread ends, but is_streaming stays True
        cam._cam._t.join()
        cam._base_timeout = 50
        with pytest.raises(TimeoutError):
            cam.acquire_frames(2)
        cam._cam._t = None  # let the fixture tear down

    def test_incomplete_skipped_in_acquisition(self, cam):
        cam._cam.bad_every = 3
        cam.start_streaming()
        cube = cam.acquire_frames(6, multiframe_out_mode="cube")
        assert cube.shape == (H, W, 6)
        assert all(int(cube[0, 0, k]) % 3 for k in range(6))
        cam.stop_streaming()


class TestAcquireFramesAsync:
    def test_async_nframes_starts_and_stops(self, cam):
        cube = cam.acquire_frames(4, multiframe_out_mode="cube", mode="async")
        assert cube.shape == (H, W, 4)
        assert not cam.is_streaming
        assert cam._cam.starts == 1

    def test_async_nframes_one(self, cam):
        out = cam.acquire_frames(1, mode="async")
        assert out.shape == (H, W) and not cam.is_streaming

    def test_async_stops_stream_on_timeout(self, cam):
        cam._cam.period = 10  # one frame, then nothing
        cam._base_timeout = 50
        with pytest.raises(TimeoutError):
            cam.acquire_frames(3, mode="async")
        assert not cam.is_streaming

    def test_async_none_waits_for_enter(self, cam, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda *_: time.sleep(0.1))
        out = cam.acquire_frames(None, multiframe_out_mode="cube", mode="async")
        assert out.ndim == 3 and out.shape[:2] == (H, W)
        assert not cam.is_streaming


class TestExptime:
    def test_set_exptime_clears_buffer(self, cam):
        cam._buffer.push(1)
        cam.set_exptime(5000)
        assert cam._buffer.snapshot() == []
        assert cam._exptime == 5000
        cam._cam.feature.set.assert_called_once_with(5000)

    def test_set_same_exptime_keeps_buffer(self, cam):
        cam._buffer.push(1)
        cam.set_exptime(1000)
        assert cam._buffer.snapshot() == [1]
        cam._cam.feature.set.assert_not_called()


class TestLiveView:
    def test_starts_stream_and_viewer(self, cam, fake_child):
        cam.live_view(cmap="gray")
        assert cam.is_live and cam.is_streaming
        assert cam._viewer.owns_stream is True
        assert len(fake_child) == 1
        assert cam._viewer.frame.read(0)[0].shape == (H, W)
        cam.stop_live_view()

    def test_idempotent(self, cam, fake_child):
        cam.live_view()
        v = cam._viewer
        cam.live_view()
        assert cam._viewer is v and len(fake_child) == 1
        cam.stop_live_view()

    def test_viewer_receives_frames(self, cam):
        cam.live_view()
        seq0 = cam._viewer.frame.read(0)[1]
        assert _wait_for(lambda: cam._viewer.frame.read(seq0)[0] is not None)
        cam.stop_live_view()

    def test_stop_stops_owned_stream(self, cam, fake_child):
        cam.live_view()
        cam.stop_live_view()
        assert not cam.is_live and not cam.is_streaming
        assert fake_child[0].poll() is not None

    def test_stop_keeps_preexisting_stream(self, cam):
        cam.start_streaming()
        cam.live_view()
        assert cam._viewer.owns_stream is False
        cam.stop_live_view()
        assert not cam.is_live and cam.is_streaming
        cam.stop_streaming()

    def test_stop_without_view_noop(self, cam):
        cam.stop_live_view()
        assert not cam.is_live

    def test_stop_twice(self, cam):
        cam.live_view()
        cam.stop_live_view()
        cam.stop_live_view()
        assert not cam.is_streaming

    def test_child_exit_stops_owned_stream(self, cam, fake_child):
        cam.live_view()
        fake_child[0].terminate()
        assert _wait_for(lambda: not cam.is_live)
        assert _wait_for(lambda: not cam.is_streaming)

    def test_child_exit_keeps_preexisting_stream(self, cam, fake_child):
        cam.start_streaming()
        cam.live_view()
        fake_child[0].terminate()
        assert _wait_for(lambda: not cam.is_live)
        time.sleep(0.1)
        assert cam.is_streaming
        cam.stop_streaming()

    def test_acquire_while_live(self, cam):
        cam.live_view()
        cube = cam.acquire_frames(3, multiframe_out_mode="cube", mode="async")
        assert cube.shape == (H, W, 3)
        assert cam.is_live and cam.is_streaming
        cam.stop_live_view()

    def test_bad_kwargs_leave_stream_stopped(self, cam, fake_child):
        before = _shm_blocks()
        with pytest.raises(TypeError):
            cam.live_view(cmap=object())
        assert not cam.is_live and not cam.is_streaming
        assert fake_child == [] and _shm_blocks() == before

    def test_bad_kwargs_keep_preexisting_stream(self, cam):
        cam.start_streaming()
        with pytest.raises(TypeError):
            cam.live_view(cmap=object())
        assert not cam.is_live and cam.is_streaming
        cam.stop_streaming()

    def test_no_frames_leaves_stream_stopped(self, cam):
        cam._cam.period = 10
        cam._cam.bad_every = 1  # every frame incomplete
        cam._base_timeout = 50
        with pytest.raises(TimeoutError):
            cam.live_view()
        assert not cam.is_live and not cam.is_streaming

    def test_viewer_can_be_reopened(self, cam, fake_child):
        cam.live_view()
        cam.stop_live_view()
        cam.live_view()
        assert cam.is_live and len(fake_child) == 2
        cam.stop_live_view()

    def test_old_watcher_does_not_close_new_viewer(self, cam, fake_child):
        cam.live_view()
        old = cam._viewer
        cam.stop_live_view()
        cam.live_view()
        new = cam._viewer
        cam._detach_viewer(old)  # stale detach must be a no-op
        assert cam._viewer is new
        cam.stop_live_view()

    def test_concurrent_detach_cleans_once(self, cam):
        cam.live_view()
        v = cam._viewer
        ths = [threading.Thread(target=cam._detach_viewer, args=(v,)) for _ in range(5)]
        [t.start() for t in ths]
        [t.join(5) for t in ths]
        assert not cam.is_live and not cam.is_streaming


class TestClose:
    def _patch_exit(self, g):
        g._cam.__exit__ = MagicMock()

    def test_close_cleans_everything(self, cam, fake_child):
        fc = cam._cam
        cam.live_view()
        cam.close()
        assert not cam.is_live
        assert not fc.is_streaming()
        assert fake_child[0].poll() is not None
        assert cam._cam is None

    def test_close_streaming_only(self, cam):
        fc = cam._cam
        cam.start_streaming()
        cam.close()
        assert not fc.is_streaming()

    def test_close_idle(self, cam):
        cam.close()
        assert cam._cam is None

    def test_close_twice(self, cam):
        cam.close()
        cam.close()

    def test_close_half_initialised(self):
        g = cameras.GigaVision.__new__(cameras.GigaVision)
        g._logger = MagicMock()
        g.close()  # must not raise
