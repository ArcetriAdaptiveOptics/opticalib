"""
Kernel bridge of CalpyGUI
=========================

The IPython session of CalpyGUI runs in a **separate kernel process**
(``ipykernel``), so the window stays responsive while commands run, commands
can be interrupted, and a crashing hardware SDK cannot take the GUI down.

:class:`KernelBridge` owns the kernel and the console widget and is the only
way the rest of the GUI talks to the kernel:

* :meth:`KernelBridge.run` queues code that is echoed and executed in the
  console, as if typed by the user, and reports its progress and outcome
  through a :class:`Task`;
* :meth:`KernelBridge.query` evaluates expressions silently (nothing shown in
  the console) and returns their values, e.g. the workspace listing.

Figures and arrays published by :mod:`opticalib.gui.kernel_side` are
intercepted by :class:`CalpyConsole` and re-emitted as Qt signals.
"""

import ast
import base64
import itertools
import json
import os
import re
import sys
import time
import traceback
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from qtpy.QtCore import QObject, QTimer, Signal
from qtconsole.manager import QtKernelManager
from qtconsole.rich_jupyter_widget import RichJupyterWidget

from .kernel_side import FIGURE_MIME, IMAGE_MIME

#: Matplotlib backend selected in the kernel process.
KERNEL_MPL_BACKEND = "module://opticalib.gui.kernel_backend"

#: Expression reaching the kernel-side helpers even if _gui was deleted
#: (e.g. by %reset).
KERNEL_SIDE = "__import__('sys').modules['opticalib.gui.kernel_side']"

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_FRACTION_RE = re.compile(r"(\d+)\s*/\s*(\d+)")
_PERCENT_RE = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*%\|")


def strip_ansi(text: str) -> str:
    """
    Remove ANSI escape sequences (colors) from *text*.

    Parameters
    ----------
    text : str
        Text possibly containing terminal escape codes.

    Returns
    -------
    str
        Plain text.
    """
    return _ANSI_RE.sub("", text)


def parse_progress(text: str) -> Tuple[Optional[float], str]:
    """
    Extract progress information from a chunk of console output.

    Recognises ``tqdm`` bars (``42%|####``) and counters such as ``12/88``,
    which opticalib prints with carriage returns while building simulated
    devices or running command histories.

    Parameters
    ----------
    text : str
        Output chunk (stdout or stderr).

    Returns
    -------
    fraction : float or None
        Completed fraction in ``[0, 1]``, or ``None`` if not found.
    detail : str
        The last non-empty output line, for display.
    """
    lines = [l.strip() for l in re.split(r"[\r\n]+", strip_ansi(text)) if l.strip()]
    detail = lines[-1] if lines else ""
    for line in reversed(lines):
        match = _PERCENT_RE.search(line)
        if match:
            return min(float(match.group(1)) / 100.0, 1.0), detail
        match = _FRACTION_RE.fullmatch(line) or _FRACTION_RE.search(line)
        if match:
            done, total = int(match.group(1)), int(match.group(2))
            if total > 0 and 0 <= done <= total:
                return done / total, detail
    return None, detail


def _kernel_argv() -> List[str]:
    """Command line of the kernel process (same interpreter as the GUI)."""
    return [
        sys.executable,
        "-Xfrozen_modules=off",
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]


class Task(QObject):
    """
    A piece of code queued for execution in the console.

    Attributes
    ----------
    title : str
        Human-readable description shown in the activity panel.
    code : str
        The Python / IPython source.
    state : str
        ``'queued'``, ``'running'``, ``'done'``, ``'error'`` or ``'cancelled'``.
    progress : float or None
        Completed fraction when known.
    detail : str
        Last output line of the running code.
    error : dict or None
        ``{'ename', 'evalue', 'traceback'}`` when the code failed.
    quiet : bool
        Whether the task is hidden from the activity panel.

    Signals
    -------
    changed
        Emitted whenever the state, progress or detail changes.
    finished
        Emitted once, when the task reaches a final state.
    """

    changed = Signal()
    finished = Signal()

    _ids = itertools.count(1)

    def __init__(
        self,
        code: str,
        title: Optional[str] = None,
        on_done: Optional[Callable[["Task"], None]] = None,
        on_error: Optional[Callable[["Task"], None]] = None,
        quiet: bool = False,
    ) -> None:
        """Create a queued task."""
        super().__init__()
        self.id = next(self._ids)
        self.code = code
        lines = code.strip().splitlines()
        self.title = title or (lines[-1][:80] if lines else "(empty)")
        self.on_done = on_done
        self.on_error = on_error
        self.quiet = quiet
        self.state = "queued"
        self.progress: Optional[float] = None
        self.detail = ""
        self.error: Optional[Dict[str, Any]] = None
        self.msg_id: Optional[str] = None
        self.queued_at = time.monotonic()
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self.bootstrap = False
        self._saved_input = ""

    @property
    def elapsed(self) -> float:
        """Seconds spent running (0 while queued)."""
        if self.started_at is None:
            return 0.0
        end = self.finished_at if self.finished_at is not None else time.monotonic()
        return end - self.started_at

    @property
    def is_final(self) -> bool:
        """Whether the task has finished (successfully or not)."""
        return self.state in ("done", "error", "cancelled")

    def _start(self, msg_id: str) -> None:
        self.msg_id = msg_id
        self.state = "running"
        self.started_at = time.monotonic()
        self.changed.emit()

    def _update_output(self, text: str) -> None:
        fraction, detail = parse_progress(text)
        if fraction is None and not detail:
            return
        if fraction is not None:
            self.progress = fraction
        if detail:
            self.detail = detail[:120]
        self.changed.emit()

    def _finish(self, state: str, error: Optional[Dict[str, Any]] = None) -> None:
        if self.is_final:
            return
        self.state = state
        self.error = error
        self.finished_at = time.monotonic()
        if self.started_at is None:
            self.started_at = self.finished_at
        if state == "done":
            self.progress = 1.0
        self.changed.emit()
        self.finished.emit()
        callback = self.on_done if state == "done" else self.on_error
        if callback is not None:
            _safe_call(callback, self)


class CalpyConsole(RichJupyterWidget):
    """
    qtconsole widget that routes the CalpyGUI outputs to the GUI.

    Signals
    -------
    figure_received(dict)
        A matplotlib figure published by the kernel (``uid``, ``num``,
        ``title``, ``png`` bytes).
    image_received(dict)
        An array sent to the viewer (``path``, ``title``, ``shape``,
        ``dtype``).
    request_sent(str, bool)
        An execute request was sent (``msg_id``, ``hidden``).
    """

    figure_received = Signal(dict)
    image_received = Signal(dict)
    request_sent = Signal(str, bool)

    def _execute(self, source, hidden):
        """Reimplemented to report the ``msg_id`` of every request."""
        before = set(self._request_info["execute"])
        super()._execute(source, hidden)
        for msg_id in list(self._request_info["execute"]):
            if msg_id not in before:
                self.request_sent.emit(msg_id, hidden)

    def _handle_display_data(self, msg):
        """Reimplemented to intercept figures and arrays for the GUI."""
        data = msg.get("content", {}).get("data", {})
        if FIGURE_MIME in data:
            payload = dict(data[FIGURE_MIME])
            payload["png"] = base64.b64decode(payload.get("png", ""))
            self.figure_received.emit(payload)
            return
        if IMAGE_MIME in data:
            self.image_received.emit(dict(data[IMAGE_MIME]))
            return
        super()._handle_display_data(msg)


class _Query:
    """A silent request waiting for its reply."""

    __slots__ = ("callback", "expressions", "code", "retried", "sent_at")

    def __init__(self, callback, expressions, code) -> None:
        self.callback = callback
        self.expressions = expressions
        self.code = code
        self.retried = False
        self.sent_at = 0.0


def _safe_call(callback: Callable, *args: Any) -> None:
    """Call *callback*, reporting (not propagating) its exceptions."""
    try:
        callback(*args)
    except Exception:
        traceback.print_exc()


class KernelBridge(QObject):
    """
    Owner of the kernel process and the console widget.

    Parameters
    ----------
    config_path : str
        Configuration file exported to the kernel as ``AOCONF``.
    init_file : str or None
        The calpy bootstrap script (``initCalpy.py``) run at startup.
    tmp_dir : str
        Folder used by the kernel to hand arrays over to the GUI.
    parent : QObject, optional
        Parent object.

    Signals
    -------
    state_changed(str)
        Kernel state: ``'starting'``, ``'idle'``, ``'busy'``, ``'restarting'``
        or ``'dead'``.
    bootstrap_step(str, str, str)
        Startup progress: step key, status (``'running'``, ``'done'``,
        ``'error'``) and message.
    ready()
        The kernel is bootstrapped and accepts GUI tasks.
    task_added(object)
        A :class:`Task` was queued.
    execution_finished()
        Any console execution (GUI task or typed by the user) finished.
    restart_requested()
        The user asked the console to restart the kernel (Ctrl+.); the
        application should confirm and call :meth:`restart`.
    kernel_died(str)
        The kernel died and could not be restarted automatically.

    Notes
    -----
    Completion of console executions is detected from the kernel's
    ``execute_reply`` messages, not from the console widget, whose request
    bookkeeping is reset on startup and restarts.
    """

    state_changed = Signal(str)
    bootstrap_step = Signal(str, str, str)
    ready = Signal()
    task_added = Signal(object)
    execution_finished = Signal()
    restart_requested = Signal()
    kernel_died = Signal(str)

    #: Startup steps, as (key, label).
    BOOTSTRAP_STEPS = [
        ("kernel", "Starting the Python kernel"),
        ("opticalib", "Importing opticalib"),
        ("calpy", "Loading the calpy environment"),
    ]

    #: Seconds to wait for the kernel to answer before reporting a failure.
    STARTUP_TIMEOUT = 120.0
    #: Seconds an unacknowledged request may wait while the kernel is idle
    #: before the connection is considered broken (see :meth:`_watchdog_check`).
    REQUEST_STALL = 8.0

    def __init__(
        self,
        config_path: str,
        init_file: Optional[str],
        tmp_dir: str,
        parent: Optional[QObject] = None,
    ) -> None:
        """Create the console widget; the kernel starts with :meth:`start`."""
        super().__init__(parent)
        self._config_path = config_path
        self._init_file = init_file
        self._tmp_dir = tmp_dir

        self.console = CalpyConsole()
        # Restarts go through the application (confirmation + bootstrap).
        self.console.custom_restart = True
        self.console.custom_restart_requested.connect(self.restart_requested.emit)
        self.console.request_sent.connect(self._on_request_sent)

        self._km: Optional[QtKernelManager] = None
        self._kc = None
        self._state = "dead"
        self._ready = False
        self._bootstrapping = False
        self._generation = 0
        self._waiting_kernel_info = False
        self._kernel_info_since = 0.0
        self._queue: Deque[Task] = deque()
        self._dispatching: Optional[Task] = None
        self._tasks_by_msg: Dict[str, Task] = {}
        self._inflight_visible: set = set()
        self._queries: Dict[str, _Query] = {}
        self._idle_timer = QTimer(self)
        self._idle_timer.setSingleShot(True)
        self._idle_timer.setInterval(120)
        self._idle_timer.timeout.connect(self._on_idle_timeout)
        self._kernel_info_timer = QTimer(self)
        self._kernel_info_timer.setInterval(1000)
        self._kernel_info_timer.timeout.connect(self._resend_kernel_info)
        # Requests acknowledged by the kernel (seen as parents on iopub).
        self._acknowledged: set = set()
        self._visible_sent_at: Dict[str, float] = {}
        self._reconnects: List[float] = []
        self._watchdog = QTimer(self)
        self._watchdog.setInterval(2000)
        self._watchdog.timeout.connect(self._watchdog_check)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> str:
        """The current kernel state."""
        return self._state

    @property
    def is_ready(self) -> bool:
        """Whether the kernel is bootstrapped."""
        return self._ready

    @property
    def kernel_manager(self) -> Optional[QtKernelManager]:
        """The jupyter kernel manager (``None`` before :meth:`start`)."""
        return self._km

    @property
    def kernel_pid(self) -> Optional[int]:
        """Process ID of the running kernel (``None`` if not running)."""
        process = getattr(getattr(self._km, "provisioner", None), "process", None)
        pid = getattr(process, "pid", None)
        return pid if isinstance(pid, int) else None

    @property
    def pending_tasks(self) -> List[Task]:
        """Queued and running GUI tasks."""
        running = [t for t in self._tasks_by_msg.values() if not t.is_final]
        return running + [t for t in self._queue if not t.bootstrap]

    @property
    def is_busy(self) -> bool:
        """Whether code (GUI task or typed by the user) is running."""
        return bool(self._inflight_visible) or self._state == "busy"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def config_path(self) -> str:
        """The configuration file exported to the kernel as ``AOCONF``."""
        return self._config_path

    def set_config_path(self, config_path: str) -> None:
        """
        Use another configuration file for the kernel environment.

        Updates ``AOCONF`` for the next (re)starts of the kernel; the running
        kernel is switched separately, with
        ``opticalib.set_configuration_file``.

        Parameters
        ----------
        config_path : str
            The configuration file.
        """
        self._config_path = config_path
        if self._km is not None:
            try:
                self._km.update_env(env={"AOCONF": config_path})
            except AttributeError:  # jupyter_client < 8.5
                launch = getattr(self._km, "_launch_args", None)
                if isinstance(launch, dict) and isinstance(launch.get("env"), dict):
                    launch["env"]["AOCONF"] = config_path

    def kernel_env(self) -> Dict[str, str]:
        """
        Return the environment of the kernel process.

        Returns
        -------
        dict
            A copy of the GUI environment with ``AOCONF``, ``MPLBACKEND`` and
            ``CALPY_GUI_TMP`` set for the kernel.
        """
        env = dict(os.environ)
        env["AOCONF"] = self._config_path
        env["MPLBACKEND"] = KERNEL_MPL_BACKEND
        env["CALPY_GUI_TMP"] = self._tmp_dir
        env["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
        return env

    def start(self) -> None:
        """Start the kernel process and connect the console to it."""
        from jupyter_client.kernelspec import KernelSpec

        self._set_state("starting")
        self.bootstrap_step.emit("kernel", "running", "")
        km = QtKernelManager()
        # Always run the kernel with the GUI's interpreter, whatever
        # kernelspecs are installed for the user.
        km._kernel_spec = KernelSpec(
            argv=_kernel_argv(),
            display_name="CalpyGUI (opticalib)",
            language="python",
        )
        self._configure_transport(km)
        km.kernel_restarted.connect(self._on_kernel_auto_restarted)
        try:
            km.start_kernel(env=self.kernel_env())
        except Exception as exc:
            self._set_state("dead")
            self.bootstrap_step.emit("kernel", "error", f"The kernel could not start: {exc}")
            return
        km.add_restart_callback(self._on_kernel_dead, "dead")
        self._km = km
        self.console.kernel_manager = km
        self._connect_client()
        self._watchdog.start()
        self._begin_bootstrap()

    def _connect_client(self) -> None:
        """
        (Re)create the kernel client and connect it to the bridge and console.

        A fresh client (new sockets) is used after every restart: the old
        sockets can keep a stale connection to the dead kernel for a while,
        and requests routed to it would never be answered.
        """
        old = self._kc
        if old is not None:
            try:
                old.stop_channels()
            except Exception:
                pass
        kc = self._km.client()
        kc.start_channels()
        kc.iopub_channel.message_received.connect(self._on_iopub)
        kc.shell_channel.message_received.connect(self._on_shell)
        self._kc = kc
        self.console.kernel_client = kc

    def _configure_transport(self, km: QtKernelManager) -> None:
        """
        Use local IPC sockets instead of TCP when possible.

        IPC sockets are only reachable by the local user, whereas the default
        TCP transport is unencrypted.  Unix socket paths are limited to about
        104 bytes, so TCP is kept when the temporary folder path is too long.
        """
        if os.name != "posix":
            return
        prefix = os.path.join(self._tmp_dir, "kernel")
        if len(prefix) > 80:
            return
        km.transport = "ipc"
        km.ip = prefix

    def shutdown(self, now: bool = False) -> None:
        """
        Stop the kernel process (called when the window closes).

        Parameters
        ----------
        now : bool, optional
            Kill the kernel immediately instead of asking it to exit.
        """
        self._kernel_info_timer.stop()
        self._watchdog.stop()
        self._generation += 1
        self._fail_pending("The kernel was shut down.")
        if self._kc is not None:
            try:
                self._kc.stop_channels()
            except Exception:
                pass
        if self._km is not None and self._km.has_kernel:
            try:
                self._km.shutdown_kernel(now=now)
            except Exception:
                pass
        self._ready = False
        self._set_state("dead")

    def interrupt(self) -> None:
        """Interrupt the code running in the kernel (like Ctrl+C)."""
        if self._km is not None and self._km.has_kernel:
            self._km.interrupt_kernel()

    def restart(self) -> None:
        """Restart the kernel and run the calpy bootstrap again."""
        if self._km is None:
            return
        self._generation += 1
        self._fail_pending("The kernel was restarted.")
        self._set_state("restarting")
        try:
            self._km.restart_kernel(now=True)
        except Exception as exc:
            self._set_state("dead")
            self.bootstrap_step.emit("kernel", "error", f"The kernel could not restart: {exc}")
            return
        self.console.reset(clear=True)
        self._connect_client()
        self._begin_bootstrap()

    # ------------------------------------------------------------------
    # Execution API
    # ------------------------------------------------------------------

    def run(
        self,
        code: str,
        title: Optional[str] = None,
        on_done: Optional[Callable[[Task], None]] = None,
        on_error: Optional[Callable[[Task], None]] = None,
        quiet: bool = False,
    ) -> Task:
        """
        Queue *code* for execution in the console.

        The code is echoed in the console as if typed by the user and runs as
        soon as the console is free.  Text the user was typing is restored
        afterwards.

        Parameters
        ----------
        code : str
            Python / IPython source.
        title : str, optional
            Description shown in the activity panel.
        on_done, on_error : callable, optional
            Called with the :class:`Task` when it succeeds or fails.
        quiet : bool, optional
            Hide the task from the activity panel.

        Returns
        -------
        Task
            The queued task.
        """
        task = Task(code, title, on_done=on_done, on_error=on_error, quiet=quiet)
        self._queue.append(task)
        self.task_added.emit(task)
        QTimer.singleShot(0, self._dispatch)
        return task

    def cancel(self, task: Task) -> bool:
        """
        Remove a queued task before it starts.

        Parameters
        ----------
        task : Task
            The task to cancel.

        Returns
        -------
        bool
            ``False`` when the task already started (use :meth:`interrupt`).
        """
        if task in self._queue:
            self._queue.remove(task)
            task._finish("cancelled")
            return True
        return False

    def query(
        self,
        expressions: Dict[str, str],
        callback: Callable[[Dict[str, Any]], None],
        code: str = "",
    ) -> None:
        """
        Silently run *code* and evaluate *expressions* in the kernel.

        Nothing is shown in the console.  The *callback* receives a dict
        mapping each expression key to its value (parsed from JSON when the
        expression returns a JSON string), or to an ``Exception`` instance
        when the evaluation failed (also when the kernel is restarted or
        shut down before replying).

        Parameters
        ----------
        expressions : dict
            Mapping of keys to Python expressions.
        callback : callable
            Receives the results dict.
        code : str, optional
            Statements executed before evaluating the expressions.
        """
        if self._kc is None:
            _safe_call(callback, {k: RuntimeError("The kernel is not running.") for k in expressions})
            return
        self._send_query(_Query(callback, expressions, code))

    def _send_query(self, query: _Query) -> None:
        # stop_on_error=False: a failing query never aborts other requests.
        msg_id = self._kc.execute(
            query.code,
            silent=True,
            store_history=False,
            user_expressions=query.expressions,
            allow_stdin=False,
            stop_on_error=False,
        )
        query.sent_at = time.monotonic()
        self._queries[msg_id] = query

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    def _current(self, callback: Callable) -> Callable:
        """Wrap *callback* so it is ignored after a restart or shutdown."""
        generation = self._generation

        def guarded(*args):
            if generation == self._generation:
                callback(*args)

        return guarded

    def _begin_bootstrap(self) -> None:
        self._generation += 1
        self._ready = False
        self._bootstrapping = True
        self._queue = deque(t for t in self._queue if not t.bootstrap)
        self._waiting_kernel_info = True
        self._kernel_info_since = time.monotonic()
        self._kc.kernel_info()
        self._kernel_info_timer.start()

    def _resend_kernel_info(self) -> None:
        """Ask again until the kernel answers (requests sent too early are lost)."""
        if not self._waiting_kernel_info or self._kc is None:
            self._kernel_info_timer.stop()
            return
        if time.monotonic() - self._kernel_info_since > self.STARTUP_TIMEOUT:
            self._kernel_info_timer.stop()
            self._waiting_kernel_info = False
            self._bootstrapping = False
            self._set_state("dead")
            self.bootstrap_step.emit(
                "kernel", "error",
                f"The kernel did not answer within {self.STARTUP_TIMEOUT:.0f} s. "
                "Try Kernel → Restart.",
            )
            return
        self._kc.kernel_info()

    def _on_kernel_info(self) -> None:
        if not self._waiting_kernel_info:
            return
        # The console resets its request bookkeeping when it receives its own
        # first kernel_info reply: bootstrap only once it has done so.
        if getattr(self.console, "_starting", False):
            QTimer.singleShot(50, self._on_kernel_info)
            return
        self._waiting_kernel_info = False
        self._kernel_info_timer.stop()
        self.bootstrap_step.emit("kernel", "done", "")
        self.bootstrap_step.emit("opticalib", "running", "")
        self.query(
            {"ok": "True"},
            self._current(self._on_install_done),
            code=(
                "import opticalib.gui.kernel_side as _calpy_ks\n"
                "_calpy_ks.install()\n"
                "del _calpy_ks\n"
            ),
        )

    def _on_install_done(self, results: Dict[str, Any]) -> None:
        if results.get("ok") is not True:
            self._bootstrap_failed("opticalib", results.get("ok", "no reply from the kernel"))
            return
        self.bootstrap_step.emit("opticalib", "done", "")
        self.bootstrap_step.emit("calpy", "running", "")
        if self._init_file is None:
            self.bootstrap_step.emit(
                "calpy", "error", "initCalpy.py not found; calpy aliases not loaded."
            )
            self._finish_bootstrap()
            return
        # safe_execfile instead of %run: no quoting or $-expansion issues
        # with the path (Windows paths, spaces, ...).
        task = Task(
            f"get_ipython().safe_execfile({self._init_file!r}, get_ipython().user_ns, raise_exceptions=True)",
            "Loading the calpy environment",
            on_done=self._current(lambda t: self._on_init_done()),
            on_error=self._current(lambda t: self._bootstrap_failed("calpy", t.error)),
            quiet=True,
        )
        task.bootstrap = True
        self._queue.appendleft(task)
        self._dispatch()

    def _on_init_done(self) -> None:
        self.bootstrap_step.emit("calpy", "done", "")
        self._finish_bootstrap()

    def _bootstrap_failed(self, step: str, error: Any) -> None:
        if isinstance(error, dict):
            message = f"{error.get('ename', 'Error')}: {error.get('evalue', '')}"
        else:
            message = str(error)
        self.bootstrap_step.emit(step, "error", message)
        self._finish_bootstrap()

    def _finish_bootstrap(self) -> None:
        self.query({"ok": "True"}, lambda r: None, code=f"{KERNEL_SIDE}.mark_baseline()")
        self._bootstrapping = False
        self._ready = True
        self.ready.emit()
        self.execution_finished.emit()
        self._dispatch()

    # ------------------------------------------------------------------
    # Dispatching
    # ------------------------------------------------------------------

    def _dispatch(self) -> None:
        """Send the next queued task to the console if it is free."""
        if not self._queue or self._inflight_visible or self._kc is None:
            return
        if self._state in ("dead", "restarting"):
            return
        task = self._queue[0]
        if not self._ready and not task.bootstrap:
            return
        self._queue.popleft()
        task._saved_input = self.console.input_buffer
        self._dispatching = task
        try:
            self.console.execute(task.code)
        except Exception as exc:
            task._finish("error", {"ename": type(exc).__name__, "evalue": str(exc), "traceback": []})
        finally:
            self._dispatching = None
        if task.msg_id is None and not task.is_final:  # the console refused it
            task._finish("error", {"ename": "RuntimeError", "evalue": "not executed", "traceback": []})
            QTimer.singleShot(0, self._dispatch)

    def _on_request_sent(self, msg_id: str, hidden: bool) -> None:
        if hidden:
            return
        self._inflight_visible.add(msg_id)
        self._visible_sent_at[msg_id] = time.monotonic()
        task = self._dispatching
        if task is not None and task.msg_id is None:
            self._tasks_by_msg[msg_id] = task
            task._start(msg_id)

    def _on_execute_reply(self, msg_id: str, content: Dict[str, Any]) -> None:
        """A visible execution finished (GUI task or typed by the user)."""
        self._inflight_visible.discard(msg_id)
        self._visible_sent_at.pop(msg_id, None)
        self._acknowledged.discard(msg_id)
        task = self._tasks_by_msg.pop(msg_id, None)
        saved = ""
        try:
            if task is not None:
                saved = task._saved_input
                if content.get("status") == "ok":
                    task._finish("done")
                else:
                    task._finish(
                        "error",
                        {
                            "ename": content.get("ename", "Aborted"),
                            "evalue": content.get("evalue", "execution aborted"),
                            "traceback": content.get("traceback", []),
                        },
                    )
        finally:
            self.execution_finished.emit()
            # After the console has shown its next prompt.
            QTimer.singleShot(0, lambda: self._after_execution(saved))

    def _after_execution(self, saved_input: str) -> None:
        if saved_input and not self._inflight_visible:
            self.console.input_buffer = saved_input
        self._dispatch()

    def _fail_pending(self, reason: str) -> None:
        error = {"ename": "KernelRestarted", "evalue": reason, "traceback": []}
        tasks = list(self._tasks_by_msg.values())
        queries = list(self._queries.values())
        self._tasks_by_msg.clear()
        self._inflight_visible.clear()
        self._visible_sent_at.clear()
        self._acknowledged.clear()
        self._queries.clear()
        for task in tasks:
            task._finish("error", error)
        for query in queries:
            keys = list(query.expressions) or ["ok"]
            _safe_call(query.callback, {k: RuntimeError(reason) for k in keys})
        # Queued tasks survive a restart: they run on the new kernel.

    # ------------------------------------------------------------------
    # Kernel messages
    # ------------------------------------------------------------------

    def _set_state(self, state: str) -> None:
        if state != "idle":
            self._idle_timer.stop()
        if state != self._state:
            self._state = state
            self.state_changed.emit(state)

    def _on_idle_timeout(self) -> None:
        if self._state != "dead":
            self._set_state("idle")

    def _on_iopub(self, msg: Dict[str, Any]) -> None:
        msg_type = msg.get("header", {}).get("msg_type")
        parent = msg.get("parent_header", {}).get("msg_id")
        if parent and (parent in self._queries or parent in self._inflight_visible):
            self._acknowledged.add(parent)
        if msg_type == "status":
            execution_state = msg["content"].get("execution_state")
            if execution_state == "busy" and self._state != "dead":
                self._set_state("busy")
            elif execution_state == "idle" and self._state != "dead":
                self._idle_timer.start()
        elif msg_type == "stream":
            task = self._tasks_by_msg.get(parent)
            if task is not None:
                task._update_output(msg["content"].get("text", ""))

    def _on_shell(self, msg: Dict[str, Any]) -> None:
        msg_type = msg.get("header", {}).get("msg_type")
        parent = msg.get("parent_header", {}).get("msg_id")
        if msg_type == "kernel_info_reply":
            self._on_kernel_info()
            return
        if msg_type != "execute_reply":
            return
        content = msg.get("content", {})
        if parent in self._queries:
            self._on_query_reply(parent, content)
        elif parent in self._inflight_visible or parent in self._tasks_by_msg:
            self._on_execute_reply(parent, content)

    def _on_query_reply(self, msg_id: str, content: Dict[str, Any]) -> None:
        query = self._queries.pop(msg_id)
        self._acknowledged.discard(msg_id)
        status = content.get("status")
        if status == "aborted" and not query.retried:
            # Aborted because a visible command failed just before: retry once.
            query.retried = True
            self._send_query(query)
            return
        if status != "ok":
            error = RuntimeError(
                f"{content.get('ename', status or 'Error')}: {content.get('evalue', '')}"
            )
            keys = list(query.expressions) or ["ok"]
            _safe_call(query.callback, {k: error for k in keys})
            return
        _safe_call(query.callback, _parse_user_expressions(content.get("user_expressions", {})))

    def _watchdog_check(self) -> None:
        """
        Detect requests lost by the connection to the kernel, and heal it.

        The kernel publishes a status message for every request it starts.
        A request that is still unacknowledged while the kernel is idle and
        nothing else runs was lost (this happens occasionally right after a
        restart).  The client is then reconnected; silent queries are sent
        again (they only read state), visible commands fail with a clear
        message instead of being executed twice.
        """
        if self._kc is None or self._state != "idle" or self._waiting_kernel_info:
            return
        now = time.monotonic()
        stalled_queries = [
            (msg_id, q) for msg_id, q in self._queries.items()
            if msg_id not in self._acknowledged and now - q.sent_at > self.REQUEST_STALL
        ]
        stalled_visible = [
            msg_id for msg_id in self._inflight_visible
            if msg_id not in self._acknowledged
            and now - self._visible_sent_at.get(msg_id, now) > self.REQUEST_STALL
        ]
        acknowledged_visible = [m for m in self._inflight_visible if m in self._acknowledged]
        if acknowledged_visible or not (stalled_queries or stalled_visible):
            return
        self._reconnects = [t for t in self._reconnects if now - t < 60] + [now]
        if len(self._reconnects) > 3:
            self._reconnects.clear()
            self._generation += 1
            self._fail_pending("The kernel stopped answering.")
            self._set_state("dead")
            self.kernel_died.emit("The kernel stopped answering. Use Kernel → Restart.")
            return
        for msg_id in stalled_visible:
            self._inflight_visible.discard(msg_id)
            self._visible_sent_at.pop(msg_id, None)
            task = self._tasks_by_msg.pop(msg_id, None)
            if task is not None:
                task._finish(
                    "error",
                    {
                        "ename": "ConnectionLost",
                        "evalue": "The command did not reach the kernel; run it again.",
                        "traceback": [],
                    },
                )
        for msg_id, _ in stalled_queries:
            self._queries.pop(msg_id, None)
        self._connect_client()
        for _, query in stalled_queries:
            self._send_query(query)
        QTimer.singleShot(0, self._dispatch)

    def _on_kernel_auto_restarted(self) -> None:
        """The kernel died and was restarted automatically."""
        self._generation += 1
        self._fail_pending("The kernel died and was restarted.")
        # Bootstrap after the console has reset itself for the new kernel.
        QTimer.singleShot(0, self._reconnect_and_bootstrap)

    def _reconnect_and_bootstrap(self) -> None:
        self._connect_client()
        self._begin_bootstrap()

    def _on_kernel_dead(self) -> None:
        """The kernel died and the automatic restarts failed."""
        reason = "The kernel died and could not be restarted."
        self._generation += 1
        self._fail_pending(reason)
        self._ready = False
        self._set_state("dead")
        if self._bootstrapping:
            self._bootstrapping = False
            self.bootstrap_step.emit("kernel", "error", reason)
        self.kernel_died.emit(reason)


def _parse_user_expressions(expressions: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert the ``user_expressions`` of an execute reply to Python values.

    Values are received as their ``repr``; string results that contain JSON
    are decoded.

    Parameters
    ----------
    expressions : dict
        The ``user_expressions`` field of an ``execute_reply``.

    Returns
    -------
    dict
        Key to value, or to an ``Exception`` when evaluation failed.
    """
    results: Dict[str, Any] = {}
    for key, reply in expressions.items():
        if reply.get("status") != "ok":
            results[key] = RuntimeError(
                f"{reply.get('ename', 'Error')}: {reply.get('evalue', '')}"
            )
            continue
        text = reply.get("data", {}).get("text/plain", "")
        try:
            value = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            results[key] = text
            continue
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                pass
        results[key] = value
    return results
