from __future__ import annotations

import importlib.util
import os
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_run_services_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "bin" / "run_services.py"
    spec = importlib.util.spec_from_file_location("test_run_services_module", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_configure_web_readiness_env_flags(monkeypatch):
    module = _load_run_services_module()
    monkeypatch.delenv("ARXIV_SANITY_READY_REQUIRE_EMBEDDING", raising=False)
    monkeypatch.delenv("ARXIV_SANITY_READY_REQUIRE_MINERU", raising=False)

    module._configure_web_readiness_env(no_embed=True, no_mineru=False)
    assert os.environ.get("ARXIV_SANITY_READY_REQUIRE_EMBEDDING") == "0"
    assert os.environ.get("ARXIV_SANITY_READY_REQUIRE_MINERU") == "1"

    # setdefault semantics: existing operator overrides are preserved.
    module._configure_web_readiness_env(no_embed=False, no_mineru=True)
    assert os.environ.get("ARXIV_SANITY_READY_REQUIRE_EMBEDDING") == "0"
    assert os.environ.get("ARXIV_SANITY_READY_REQUIRE_MINERU") == "1"


def test_build_huey_command_defaults_to_quiet():
    module = _load_run_services_module()
    cmd = module._build_huey_command(
        python_executable=sys.executable,
        consumer_script=Path("/tmp/huey_consumer.py"),
        workers=4,
        worker_type="thread",
        show_task_logs=False,
    )
    assert cmd[-1] == "-q"
    assert cmd[:3] == [sys.executable, "/tmp/huey_consumer.py", "tasks.huey"]


def test_build_huey_command_keeps_raw_logs_when_requested():
    module = _load_run_services_module()
    cmd = module._build_huey_command(
        python_executable=sys.executable,
        consumer_script=Path("/tmp/huey_consumer.py"),
        workers=2,
        worker_type="process",
        show_task_logs=True,
    )
    assert "-q" not in cmd
    assert cmd[-2:] == ["-k", "process"]


def test_summarize_task_infos_groups_terminal_and_active_counts():
    module = _load_run_services_module()
    terminal_counts, active_counts = module._summarize_task_infos(
        [
            {"model": "gpt-4o-mini", "status": "ok", "updated_time": 105.0},
            {"model": "gpt-4o-mini", "status": "failed", "updated_time": 104.0},
            {"model": "upload_parse", "status": "ok", "updated_time": 106.0},
            {"model": "upload_extract", "status": "running", "updated_time": 99.0},
            {"model": "upload_process", "status": "queued", "updated_time": 98.0},
            {"model": "gpt-4o-mini", "status": "ok", "updated_time": 80.0},
        ],
        window_start=100.0,
    )

    assert terminal_counts == {
        "summary": {"ok": 1, "failed": 1},
        "upload_parse": {"ok": 1},
    }
    assert active_counts == {
        "upload_extract": {"running": 1},
        "upload_process": {"queued": 1},
    }


def test_summarize_task_infos_excludes_window_boundary_timestamp():
    module = _load_run_services_module()

    terminal_counts, active_counts = module._summarize_task_infos(
        [
            {"model": "gpt-4o-mini", "status": "ok", "updated_time": 100.0},
            {"model": "gpt-4o-mini", "status": "failed", "updated_time": 100.1},
        ],
        window_start=100.0,
    )

    assert terminal_counts == {"summary": {"failed": 1}}
    assert active_counts == {}


def test_format_task_summary_lines_compacts_output():
    module = _load_run_services_module()
    lines = module._format_task_summary_lines(
        {
            "summary": {"ok": 3, "failed": 1},
            "upload_parse": {"ok": 2},
        },
        {
            "summary": {"queued": 5, "running": 2},
            "upload_extract": {"running": 1},
        },
        interval_s=180.0,
    )

    assert lines == [
        "[huey] past 3m: summary ok=3 fail=1 | upload_parse ok=2",
        "[huey] active: summary queued=5 running=2 | upload_extract running=1",
    ]


def test_task_summary_loop_preserves_window_after_read_failure(monkeypatch):
    module = _load_run_services_module()
    window_starts = []
    read_attempts = {"count": 0}

    class _FakeStopEvent:
        def __init__(self):
            self.calls = 0

        def wait(self, _interval):
            self.calls += 1
            return self.calls >= 3

    def _fake_read_task_infos():
        read_attempts["count"] += 1
        if read_attempts["count"] == 1:
            raise RuntimeError("temporary failure")
        return []

    def _fake_summarize(task_infos, *, window_start):
        del task_infos
        window_starts.append(window_start)
        return {}, {}

    times = iter([100.0, 110.0, 120.0])
    monkeypatch.setattr(module.time, "time", lambda: next(times))
    monkeypatch.setattr(module, "_read_task_infos", _fake_read_task_infos)
    monkeypatch.setattr(module, "_summarize_task_infos", _fake_summarize)
    monkeypatch.setattr(module, "_format_task_summary_lines", lambda *_args, **_kwargs: [])

    module._task_summary_loop(
        repo_root=Path(__file__).resolve().parents[2],
        interval_s=5.0,
        stop_event=_FakeStopEvent(),
        verbose=False,
    )

    assert window_starts == [100.0]


def test_launcher_sigterm_handler_exits_cleanly():
    module = _load_run_services_module()

    with pytest.raises(SystemExit) as excinfo:
        module._launcher_sigterm_handler(signal.SIGTERM, None)

    assert excinfo.value.code == 0


def test_parse_duration_to_ms_handles_common_units():
    module = _load_run_services_module()

    assert module._parse_duration_to_ms("85.303µs") == pytest.approx(0.085303)
    assert module._parse_duration_to_ms("12ms") == pytest.approx(12.0)
    assert module._parse_duration_to_ms("1.5s") == pytest.approx(1500.0)


def test_emit_normalized_line_summarizes_gin_access(monkeypatch):
    module = _load_run_services_module()
    printed: list[tuple[str, str]] = []

    module._LOG_STATE.clear()
    monkeypatch.setattr(module, "_SUMMARY_COUNT_THRESHOLD", 2)
    monkeypatch.setattr(module, "_SUMMARY_TIME_THRESHOLD_S", 10**9)
    monkeypatch.setattr(
        module,
        "_print_launcher_line",
        lambda prefix, message: printed.append((prefix, message)),
    )

    module._emit_normalized_line(
        "embed",
        '[GIN] 2026/03/15 - 20:34:46 | 200 | 85.303µs | 127.0.0.1 | GET "/api/version"',
    )
    module._emit_normalized_line(
        "embed",
        '[GIN] 2026/03/15 - 20:34:47 | 200 | 100µs | 127.0.0.1 | GET "/api/version"',
    )

    assert printed == [
        (
            "embed",
            "HTTP summary: GET /api/version; requests=2; avg_latency=0.09 ms; statuses=200x2",
        )
    ]


def test_emit_normalized_line_summarizes_scheduler_misfires(monkeypatch):
    module = _load_run_services_module()
    printed: list[tuple[str, str]] = []

    module._LOG_STATE.clear()
    monkeypatch.setattr(module, "_MISFIRE_SUMMARY_COUNT_THRESHOLD", 2)
    monkeypatch.setattr(module, "_MISFIRE_SUMMARY_TIME_THRESHOLD_S", 10**9)
    monkeypatch.setattr(
        module,
        "_print_launcher_line",
        lambda prefix, message: printed.append((prefix, message)),
    )

    line = (
        '[web] Run time of job "repair_stale_summary_tasks '
        '(trigger: interval[0:15:00], next run at: 2026-03-15 21:45:38 CST)" '
        "was missed by 0:00:26.345491"
    )

    module._emit_normalized_line("web", line.replace("[web] ", ""))
    module._emit_normalized_line("web", line.replace("[web] ", ""))

    assert printed == [
        (
            "web",
            "Scheduler summary: job=repair_stale_summary_tasks; last_delay=0:00:26.345491; missed_runs=2",
        )
    ]


def test_emit_normalized_line_formats_nested_prefixes(monkeypatch):
    module = _load_run_services_module()
    printed: list[tuple[str, str]] = []

    monkeypatch.setattr(
        module,
        "_print_launcher_line",
        lambda prefix, message: printed.append((prefix, message)),
    )

    module._emit_normalized_line("web", "[gunicorn] Warning: SSE enabled but worker_class=gthread")
    module._emit_normalized_line("daemon", "2026-03-15 10:00:00 | INFO | [pipeline] fetch: starting")

    assert printed == [
        ("web", "Warning: Gunicorn - SSE enabled but worker_class=gthread"),
        ("daemon", "Info: Pipeline - fetch: starting"),
    ]


def test_emit_raw_line_keeps_original_text(monkeypatch):
    module = _load_run_services_module()
    printed: list[tuple[str, str]] = []

    monkeypatch.setattr(
        module,
        "_print_launcher_line",
        lambda prefix, message: printed.append((prefix, message)),
    )

    module._emit_raw_line(
        "web",
        '[GIN] 2026/03/15 - 20:34:46 | 200 | 85.303µs | 127.0.0.1 | GET "/api/version"\n',
    )

    assert printed == [
        (
            "web",
            '[GIN] 2026/03/15 - 20:34:46 | 200 | 85.303µs | 127.0.0.1 | GET "/api/version"',
        )
    ]


def test_emit_raw_line_suppresses_opencode_session_not_found_noise(monkeypatch):
    module = _load_run_services_module()
    printed: list[tuple[str, str]] = []

    module._LOG_STATE.clear()
    monkeypatch.setattr(module, "_OPENCODE_NOISE_SUMMARY_COUNT_THRESHOLD", 2)
    monkeypatch.setattr(module, "_OPENCODE_NOISE_SUMMARY_TIME_THRESHOLD_S", 10**9)
    monkeypatch.setattr(
        module,
        "_print_launcher_line",
        lambda prefix, message: printed.append((prefix, message)),
    )

    module._emit_raw_line("opencode", "NotFoundError: NotFoundError\n")
    module._emit_raw_line("opencode", '  message: "Session not found: ses_test",\n')

    assert printed == [
        (
            "opencode",
            "Suppressed repeated OpenCode session-not-found noise; occurrences=2",
        )
    ]


def test_main_enables_raw_log_stream_flag(monkeypatch):
    module = _load_run_services_module()
    monkeypatch.setattr(module.settings.opencode, "managed", False)

    monkeypatch.setattr(
        module.argparse.ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(
            verbose=False,
            verbose_raw_logs=True,
            no_embed=True,
            no_mineru=True,
            web="none",
            with_daemon=False,
            no_huey=True,
            huey_workers=None,
            huey_worker_type=None,
            no_wait=False,
            wait_timeout=60.0,
            fetch_compute=None,
            summary_source=None,
        ),
    )

    rc = module.main()

    assert rc == 0
    assert module._USE_RAW_LOG_STREAM is True


def test_main_verbose_raw_logs_keeps_huey_task_logs(monkeypatch):
    module = _load_run_services_module()
    captured = {}
    monkeypatch.setattr(module.settings.opencode, "managed", False)

    monkeypatch.setattr(
        module.argparse.ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(
            verbose=False,
            verbose_raw_logs=True,
            no_embed=True,
            no_mineru=True,
            web="none",
            with_daemon=False,
            no_huey=False,
            huey_workers=None,
            huey_worker_type=None,
            no_wait=True,
            wait_timeout=60.0,
            fetch_compute=None,
            summary_source=None,
            task_summary_interval=0.0,
        ),
    )

    def _fake_build_huey_command(**kwargs):
        captured["show_task_logs"] = kwargs["show_task_logs"]
        return [sys.executable, "-c", "pass"]

    monkeypatch.setattr(module, "_build_huey_command", _fake_build_huey_command)
    monkeypatch.setattr(module, "_print_startup_context", lambda **_kwargs: None)
    monkeypatch.setattr(module, "_flush_log_summaries", lambda: None)
    monkeypatch.setattr(module, "_stop_process", lambda *_args, **_kwargs: None)

    class _FakeProc:
        def poll(self):
            return 0

    monkeypatch.setattr(module, "_start_service", lambda _spec: _FakeProc())

    rc = module.main()

    assert rc == 0
    assert captured["show_task_logs"] is True


def test_main_verbose_raw_logs_disables_task_summary_thread(monkeypatch):
    module = _load_run_services_module()
    thread_started = {"value": False}
    monkeypatch.setattr(module.settings.opencode, "managed", False)

    monkeypatch.setattr(
        module.argparse.ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(
            verbose=False,
            verbose_raw_logs=True,
            no_embed=True,
            no_mineru=True,
            web="none",
            with_daemon=False,
            no_huey=False,
            huey_workers=None,
            huey_worker_type=None,
            no_wait=True,
            wait_timeout=60.0,
            fetch_compute=None,
            summary_source=None,
            task_summary_interval=180.0,
        ),
    )
    monkeypatch.setattr(
        module,
        "_build_huey_command",
        lambda **_kwargs: [sys.executable, "-c", "pass"],
    )
    monkeypatch.setattr(module, "_print_startup_context", lambda **_kwargs: None)
    monkeypatch.setattr(module, "_flush_log_summaries", lambda: None)
    monkeypatch.setattr(module, "_stop_process", lambda *_args, **_kwargs: None)

    class _FakeProc:
        def poll(self):
            return 0

    class _FakeThread:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def start(self):
            thread_started["value"] = True

    monkeypatch.setattr(module, "_start_service", lambda _spec: _FakeProc())
    monkeypatch.setattr(module.threading, "Thread", _FakeThread)

    rc = module.main()

    assert rc == 0
    assert thread_started["value"] is False


def test_main_managed_opencode_uses_local_health_target(monkeypatch):
    module = _load_run_services_module()
    started_specs = []

    monkeypatch.setattr(
        module.argparse.ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(
            verbose=False,
            verbose_raw_logs=False,
            no_embed=True,
            no_mineru=True,
            web="none",
            with_daemon=False,
            no_huey=True,
            huey_workers=None,
            huey_worker_type=None,
            no_wait=True,
            wait_timeout=60.0,
            fetch_compute=None,
            summary_source=None,
            task_summary_interval=0.0,
        ),
    )
    monkeypatch.setattr(module.settings.opencode, "managed", True)
    monkeypatch.setattr(module.settings.opencode, "base_url", "https://external.example/opencode")
    monkeypatch.setattr(module.settings.opencode, "host", "127.0.0.1")
    monkeypatch.setattr(module.settings.opencode, "port", 53000)
    monkeypatch.setenv("ARXIV_SANITY_OPENCODE_BASE_URL", "https://external.example/opencode")
    monkeypatch.setattr(module, "_resolve_opencode_binary", lambda: "/tmp/opencode")
    monkeypatch.setattr(module, "_print_startup_context", lambda **_kwargs: None)
    monkeypatch.setattr(module, "_flush_log_summaries", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "_stop_process", lambda *_args, **_kwargs: None)

    class _FakeProc:
        def poll(self):
            return 0

    def fake_start_service(spec):
        started_specs.append(spec)
        return _FakeProc()

    monkeypatch.setattr(module, "_start_service", fake_start_service)

    rc = module.main()

    assert rc == 0
    opencode_specs = [spec for spec in started_specs if spec.name == "opencode"]
    assert len(opencode_specs) == 1
    assert opencode_specs[0].health_url == "http://127.0.0.1:53000/global/health"
    assert os.environ.get("ARXIV_SANITY_OPENCODE_BASE_URL") == "http://127.0.0.1:53000"


def test_request_headers_include_opencode_basic_auth(monkeypatch):
    module = _load_run_services_module()

    monkeypatch.setattr(module.settings.opencode, "username", "alice")
    monkeypatch.setattr(module.settings.opencode, "password", "secret")

    headers = module._request_headers("http://127.0.0.1:53000/global/health")

    assert headers["User-Agent"] == "arxiv-sanity-x-launcher"
    assert headers["Authorization"] == "Basic YWxpY2U6c2VjcmV0"


def test_request_headers_skip_auth_for_non_opencode_endpoint(monkeypatch):
    module = _load_run_services_module()

    monkeypatch.setattr(module.settings.opencode, "username", "alice")
    monkeypatch.setattr(module.settings.opencode, "password", "secret")

    headers = module._request_headers("http://127.0.0.1:55555/ready")

    assert headers == {"User-Agent": "arxiv-sanity-x-launcher"}


def test_main_logs_diagnostic_body_for_unready_service(monkeypatch):
    module = _load_run_services_module()
    printed = []

    monkeypatch.setattr(module.settings.opencode, "managed", False)
    monkeypatch.setattr(
        module.argparse.ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(
            verbose=False,
            verbose_raw_logs=False,
            no_embed=True,
            no_mineru=True,
            web="python",
            with_daemon=False,
            no_huey=True,
            huey_workers=None,
            huey_worker_type=None,
            no_wait=False,
            wait_timeout=1.0,
            fetch_compute=None,
            summary_source=None,
            task_summary_interval=0.0,
        ),
    )
    monkeypatch.setattr(module, "_print_startup_context", lambda **_kwargs: None)
    monkeypatch.setattr(module, "_flush_log_summaries", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "_stop_process", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "_http_get_text", lambda url, timeout_s=2.0: '{"status":"error"}')
    monkeypatch.setattr(module, "_wait_for_all_services", lambda *_args, **_kwargs: {"web": False})
    monkeypatch.setattr(
        module,
        "_print_launcher_line",
        lambda prefix, message: printed.append((prefix, message)),
    )

    class _FakeProc:
        def poll(self):
            return 0

    monkeypatch.setattr(module, "_start_service", lambda _spec: _FakeProc())

    rc = module.main()

    assert rc == 2
    assert any(prefix == "launcher" and "web diagnostic body" in message for prefix, message in printed)


def test_wait_for_all_services_uses_longer_timeout_for_web(monkeypatch):
    module = _load_run_services_module()
    calls = []

    def fake_http_ok(url, timeout_s=1.0):
        calls.append((url, timeout_s))
        return True

    monkeypatch.setattr(module, "_http_ok", fake_http_ok)

    status = module._wait_for_all_services(
        [
            ("web", "http://localhost:55555/ready"),
            ("embed", "http://localhost:54000/api/version"),
        ],
        timeout_s=1.0,
        verbose=False,
    )

    assert status == {"web": True, "embed": True}
    assert ("http://localhost:55555/ready", 5.0) in calls
    assert ("http://localhost:54000/api/version", 1.0) in calls


def test_log_opencode_models_collapses_to_aliases(monkeypatch, capsys):
    module = _load_run_services_module()

    monkeypatch.setattr(
        module,
        "_http_get_json",
        lambda url, timeout_s=2.5: {
            "providers": [
                {
                    "id": "openai",
                    "models": {
                        "gpt-5.4": {"id": "gpt-5.4"},
                        "gpt-5.4-mini": {"id": "gpt-5.4-mini"},
                    },
                },
                {
                    "id": "rightcode-openai",
                    "models": {"gpt-5.4": {"id": "gpt-5.4"}},
                },
                {
                    "id": "anthropic",
                    "models": {"claude-sonnet-4-6": {"id": "claude-sonnet-4-6"}},
                },
            ]
        },
    )

    module._log_opencode_models("http://127.0.0.1:53000", verbose=True)

    out = capsys.readouterr().out
    assert "gpt-5.4" in out
    assert "gpt-5.4-mini" in out
    assert "anthropic/claude-sonnet-4-6" in out
    assert "openai/gpt-5.4" not in out
    assert "rightcode-openai/gpt-5.4" not in out
