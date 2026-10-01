"""
Tests for the bookkeeping of the CalpyGUI kernel bridge
(opticalib.gui.kernel.KernelBridge), driven with a fake kernel client.

These cover races that are rare with a real kernel: restarts during the
bootstrap, replies lost by the console, aborted queries, exceptions in
callbacks, and the console's own restart shortcut.
"""

import itertools

import pytest

pytest.importorskip("qtconsole", exc_type=ImportError)


class FakeKernelClient:
    """Records requests; replies are injected by the tests."""

    _ids = itertools.count(1)

    def __init__(self):
        self.executed = []
        self.kernel_info_requests = 0

    def execute(self, code, silent=False, store_history=True, user_expressions=None,
                allow_stdin=None, stop_on_error=True):
        msg_id = f"m{next(self._ids)}"
        self.executed.append(
            {"id": msg_id, "code": code, "silent": silent, "expressions": user_expressions or {},
             "stop_on_error": stop_on_error}
        )
        return msg_id

    def kernel_info(self):
        self.kernel_info_requests += 1

    def history(self, *args, **kwargs):
        pass


def _reply(parent, status="ok", **content):
    content["status"] = status
    return {"header": {"msg_type": "execute_reply"}, "parent_header": {"msg_id": parent}, "content": content}


def _expr_reply(parent, **values):
    exprs = {k: {"status": "ok", "data": {"text/plain": repr(v)}} for k, v in values.items()}
    return _reply(parent, user_expressions=exprs)


@pytest.fixture
def bridge(qapp, tmp_path):
    from opticalib.gui.kernel import KernelBridge

    b = KernelBridge(str(tmp_path / "c.yaml"), str(tmp_path / "initCalpy.py"), str(tmp_path))
    kc = FakeKernelClient()
    b._kc = kc
    b.console._kernel_client = kc  # bypass the property (it wires Qt signals)
    b._set_state("idle")
    b.kc = kc
    return b


def _ready(bridge):
    bridge._ready = True
    bridge._bootstrapping = False


class TestBootstrap:
    def test_waits_for_the_console_startup(self, bridge, qt_wait):
        bridge._begin_bootstrap()
        bridge.console._starting = True  # console still handling its kernel_info
        bridge._on_kernel_info()
        assert not any("install" in r["code"] for r in bridge.kc.executed)
        bridge.console._starting = False
        assert qt_wait(lambda: any("install" in r["code"] for r in bridge.kc.executed), timeout=2)

    def test_kernel_info_is_resent(self, bridge, qt_wait):
        bridge._kernel_info_timer.setInterval(20)
        bridge._begin_bootstrap()
        assert qt_wait(lambda: bridge.kc.kernel_info_requests >= 3, timeout=2)
        bridge._on_kernel_info()
        count = bridge.kc.kernel_info_requests
        qt_wait(lambda: False, timeout=0.2)
        assert bridge.kc.kernel_info_requests == count  # stopped once answered

    def test_startup_timeout(self, bridge, qt_wait, monkeypatch):
        steps = []
        bridge.bootstrap_step.connect(lambda k, s, m: steps.append((k, s)))
        monkeypatch.setattr(type(bridge), "STARTUP_TIMEOUT", 0.05)
        bridge._kernel_info_timer.setInterval(20)
        bridge._begin_bootstrap()
        assert qt_wait(lambda: ("kernel", "error") in steps, timeout=2)
        assert bridge.state == "dead"

    def test_install_without_reply_is_a_failure(self, bridge):
        steps = []
        bridge.bootstrap_step.connect(lambda k, s, m: steps.append((k, s)))
        bridge._on_install_done({})
        assert ("opticalib", "error") in steps and bridge.is_ready

    def test_stale_bootstrap_callbacks_are_ignored(self, bridge):
        steps = []
        bridge.bootstrap_step.connect(lambda k, s, m: steps.append((k, s)))
        bridge._begin_bootstrap()
        bridge.console._starting = False
        bridge._on_kernel_info()
        install = bridge.kc.executed[-1]["id"]
        bridge._generation += 1  # e.g. a restart happened meanwhile
        bridge._on_shell(_expr_reply(install, ok=True))
        assert ("opticalib", "done") not in steps

    def test_restart_clears_queued_bootstrap(self, bridge):
        from opticalib.gui.kernel import Task

        stale = Task("old bootstrap")
        stale.bootstrap = True
        user = Task("x = 1")
        bridge._queue.extend([stale, user])
        bridge._begin_bootstrap()
        assert list(bridge._queue) == [user]

    def test_bootstrap_uses_safe_execfile(self, bridge):
        bridge._on_install_done({"ok": True})
        code = bridge.kc.executed[-1]["code"]
        assert code.startswith("get_ipython().safe_execfile(") and "%run" not in code


class TestExecution:
    def test_completion_from_the_kernel_reply(self, bridge, qt_wait):
        """A task completes even if the console forgot the request."""
        _ready(bridge)
        task = bridge.run("x = 1", "set x")
        assert qt_wait(lambda: task.state == "running", timeout=2)
        bridge.console._request_info["execute"].clear()  # console reset
        bridge._on_shell(_reply(task.msg_id))
        assert task.state == "done"
        assert not bridge._inflight_visible

    def test_error_reply(self, bridge, qt_wait):
        _ready(bridge)
        task = bridge.run("1/0")
        assert qt_wait(lambda: task.state == "running", timeout=2)
        bridge._on_shell(_reply(task.msg_id, "error", ename="ZeroDivisionError", evalue="x", traceback=[]))
        assert task.state == "error" and task.error["ename"] == "ZeroDivisionError"

    def test_callback_exception_does_not_stall_the_queue(self, bridge, qt_wait, capsys):
        _ready(bridge)

        def boom(task):
            raise RuntimeError("callback bug")

        first = bridge.run("a = 1", on_done=boom)
        second = bridge.run("b = 2")
        assert qt_wait(lambda: first.state == "running", timeout=2)
        finished = []
        bridge.execution_finished.connect(lambda: finished.append(1))
        bridge._on_shell(_reply(first.msg_id))
        assert first.state == "done" and finished == [1]
        assert qt_wait(lambda: second.state == "running", timeout=2)
        assert "callback bug" in capsys.readouterr().err

    def test_typed_commands_block_and_release_dispatch(self, bridge, qt_wait):
        _ready(bridge)
        bridge._on_request_sent("typed", False)  # the user pressed Enter
        task = bridge.run("x = 1")
        qt_wait(lambda: False, timeout=0.1)
        assert task.state == "queued"
        bridge._on_shell(_reply("typed"))
        assert qt_wait(lambda: task.state == "running", timeout=2)


class TestQueries:
    def test_queries_do_not_stop_on_error(self, bridge):
        bridge.query({"a": "1"}, lambda r: None)
        assert bridge.kc.executed[-1]["stop_on_error"] is False
        assert bridge.kc.executed[-1]["silent"] is True

    def test_aborted_query_is_retried_once(self, bridge):
        results = []
        bridge.query({"a": "1"}, results.append)
        first = bridge.kc.executed[-1]["id"]
        bridge._on_shell(_reply(first, "aborted"))
        assert results == [] and len(bridge.kc.executed) == 2
        second = bridge.kc.executed[-1]["id"]
        bridge._on_shell(_reply(second, "aborted"))
        assert isinstance(results[0]["a"], RuntimeError)

    def test_pending_queries_get_explicit_errors(self, bridge):
        results = []

        def nested(r):
            results.append(r)
            bridge.query({"again": "1"}, lambda r2: None)  # must not break the loop

        bridge.query({"a": "1", "b": "2"}, nested)
        bridge.query({"c": "3"}, results.append)
        bridge._fail_pending("restarted")
        assert [sorted(r) for r in results] == [["a", "b"], ["c"]]
        assert all(isinstance(v, RuntimeError) for r in results for v in r.values())

    def test_query_without_kernel(self, qapp, tmp_path):
        from opticalib.gui.kernel import KernelBridge

        b = KernelBridge(str(tmp_path / "c.yaml"), None, str(tmp_path))
        results = []
        b.query({"a": "1"}, results.append)
        assert isinstance(results[0]["a"], RuntimeError)


def test_console_restart_goes_through_the_app(bridge):
    requested = []
    bridge.restart_requested.connect(lambda: requested.append(1))
    bridge.console.request_restart_kernel()  # what Ctrl+. does
    assert requested == [1]


class TestWatchdog:
    """Requests lost by the kernel connection are detected and healed."""

    @pytest.fixture
    def watched(self, bridge, monkeypatch):
        monkeypatch.setattr(type(bridge), "REQUEST_STALL", 0.0)
        reconnects = []

        def fake_reconnect():
            reconnects.append(1)
            new = FakeKernelClient()
            bridge._kc = new
            bridge.kc = new

        monkeypatch.setattr(bridge, "_connect_client", fake_reconnect)
        bridge.reconnects = reconnects
        return bridge

    def test_lost_query_is_resent_on_a_new_connection(self, watched):
        results = []
        watched.query({"a": "1"}, results.append)
        watched._watchdog_check()
        assert watched.reconnects == [1]
        resent = watched.kc.executed[-1]
        assert resent["expressions"] == {"a": "1"}
        watched._on_shell(_expr_reply(resent["id"], a=1))
        assert results == [{"a": 1}]

    def test_acknowledged_query_is_left_alone(self, watched):
        watched.query({"a": "1"}, lambda r: None)
        msg_id = watched.kc.executed[-1]["id"]
        watched._on_iopub({"header": {"msg_type": "status"}, "parent_header": {"msg_id": msg_id},
                           "content": {"execution_state": "busy"}})
        watched._set_state("idle")
        watched._watchdog_check()
        assert watched.reconnects == []

    def test_busy_kernel_is_never_considered_stalled(self, watched):
        watched.query({"a": "1"}, lambda r: None)
        watched._set_state("busy")
        watched._watchdog_check()
        assert watched.reconnects == []

    def test_lost_visible_command_fails_instead_of_running_twice(self, watched, qt_wait):
        _ready(watched)
        task = watched.run("move_the_mirror()")
        assert qt_wait(lambda: task.state == "running", timeout=2)
        executed_before = len(watched.kc.executed)
        watched._watchdog_check()
        assert task.state == "error" and task.error["ename"] == "ConnectionLost"
        assert watched.reconnects == [1]
        assert all("move_the_mirror" not in r["code"] for r in watched.kc.executed[executed_before:])

    def test_gives_up_after_repeated_failures(self, watched):
        died = []
        watched.kernel_died.connect(died.append)
        for _ in range(4):
            watched._set_state("idle")
            watched.query({"a": "1"}, lambda r: None)
            watched._watchdog_check()
        assert died and watched.state == "dead"
