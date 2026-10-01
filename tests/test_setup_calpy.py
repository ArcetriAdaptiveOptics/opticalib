"""Tests for setup_calpy module.

Tests ensure the calpy entry point can locate initCalpy.py bootstrap script
both in development (source checkout) and in installed wheel packages.
"""

import os
import sys

import pytest
from pathlib import Path

import setup_calpy


class TestResolveInitFile:
    """Test initCalpy script resolution logic."""

    def test_resolve_init_script_exists_from_source(self) -> None:
        """Verify initCalpy resolves in current source environment.
        
        In development (source checkout), __init_script__/initCalpy.py exists
        at the repo root. This test ensures the fallback chain locates it.
        """
        resolved = setup_calpy._resolve_init_file()
        assert resolved is not None, (
            "calpy failed to resolve initCalpy script. "
            "Ensure opticalib/__init_script_/initCalpy.py exists (installed package) "
            "or __init_script__/initCalpy.py exists (source checkout)"
        )
        assert Path(resolved).exists(), f"Resolved path does not exist: {resolved}"
        assert "initCalpy.py" in resolved

    def test_packaged_init_script_declared(self) -> None:
        """Ensure setup.py declares packaged init script for wheels.
        
        When opticalib is installed, initCalpy.py must be shipped with it.
        This test guards against regressions where setup.py loses the
        package_data declaration for __init_script__/initCalpy.py.
        """
        setup_path = Path(__file__).resolve().parents[1] / "setup.py"
        setup_source = setup_path.read_text(encoding="utf-8")
        assert "__init_script__/initCalpy.py" in setup_source, (
            "setup.py does not declare __init_script__/initCalpy.py in package_data. "
            "This will break calpy in installed wheels."
        )

    def test_manifest_includes__init_script__(self) -> None:
        """Ensure MANIFEST.in includes packaged init script for sdists.
        
        Source distributions must include the packaged init script resource.
        """
        manifest_path = Path(__file__).resolve().parents[1] / "MANIFEST.in"
        manifest_source = manifest_path.read_text(encoding="utf-8")
        assert "__init_script__/initCalpy.py" in manifest_source, (
            "MANIFEST.in does not declare __init_script__/initCalpy.py. "
            "This will break calpy in source distributions."
        )


class TestUpdateEnvVar:
    """AOCONF must be set in-process (Windows-safe; no Unix export)."""

    def test_update_env_var_sets_aoconf(self, monkeypatch) -> None:
        monkeypatch.delenv("AOCONF", raising=False)
        target = str(Path("C:/fake/SysConfig/configuration.yaml"))
        setup_calpy.update_env_var(target)
        assert os.environ["AOCONF"] == target

    def test_prefer_sysconfig_uses_sysconfig_layout(self, tmp_path) -> None:
        exp = tmp_path / "experiment"
        sysconfig = exp / "SysConfig"
        sysconfig.mkdir(parents=True)
        cfg = sysconfig / "configuration.yaml"
        cfg.write_text("SYSTEM: {}\n", encoding="utf-8")
        candidate = str(exp / "configuration.yaml")
        assert setup_calpy._prefer_sysconfig(candidate) == str(cfg)


class TestLaunchGui:
    """``calpy --gui`` fails with a clear message instead of crashing."""

    @pytest.fixture
    def linux(self, monkeypatch):
        monkeypatch.setattr(setup_calpy.sys, "platform", "linux")
        for var in ("QT_QPA_PLATFORM", "DISPLAY", "WAYLAND_DISPLAY"):
            monkeypatch.delenv(var, raising=False)

    def test_no_display(self, linux, capsys) -> None:
        assert "no graphical display" in setup_calpy._gui_display_problem()
        with pytest.raises(SystemExit) as exit_info:
            setup_calpy._launch_gui(None)
        assert exit_info.value.code == 1
        assert "ssh -X" in capsys.readouterr().out

    @pytest.mark.parametrize("var", ["DISPLAY", "WAYLAND_DISPLAY", "QT_QPA_PLATFORM"])
    def test_display_available(self, linux, monkeypatch, var) -> None:
        monkeypatch.setenv(var, ":0")
        assert setup_calpy._gui_display_problem() is None

    def test_missing_dependencies(self, linux, monkeypatch, capsys) -> None:
        monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
        # A None entry in sys.modules makes the import raise ImportError.
        monkeypatch.setitem(sys.modules, "opticalib.gui", None)
        with pytest.raises(SystemExit) as exit_info:
            setup_calpy._launch_gui(None)
        assert exit_info.value.code == 1
        out = capsys.readouterr().out
        assert "PySide6-Essentials" in out and "PyQt5" not in out


class TestLauncher:
    """The OptiCalib desktop launcher (calpy --install-launcher)."""

    @pytest.fixture
    def experiment(self, tmp_path):
        config = tmp_path / "My Exp" / "SysConfig" / "configuration.yaml"
        config.parent.mkdir(parents=True)
        config.write_text("SYSTEM:\n  data_path: ''\n")
        return config

    @pytest.fixture
    def linux_home(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        (home / "Desktop").mkdir(parents=True)
        monkeypatch.setattr(setup_calpy.sys, "platform", "linux")
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("XDG_DATA_HOME", str(home / ".local" / "share"))
        # No desktop tools: no side effects outside the temporary home.
        monkeypatch.setattr(setup_calpy.shutil, "which", lambda name: None)
        monkeypatch.setattr(setup_calpy.Path, "home", classmethod(lambda cls: home))
        return home

    @pytest.mark.parametrize(
        "arg, quoted",
        [
            ("/plain/path", "/plain/path"),
            ("/data/My Exp/c.yaml", '"/data/My Exp/c.yaml"'),
            ('we"ird', '"we\\\\"ird"'),
            ("cost$5", '"cost\\\\$5"'),
            ("back\\slash", '"back\\\\\\\\slash"'),
            ("50%", "50%%"),
        ],
    )
    def test_exec_quoting(self, arg, quoted) -> None:
        assert setup_calpy._desktop_quote(arg) == quoted

    def test_desktop_entry(self) -> None:
        entry = setup_calpy.desktop_entry(["/env/bin/calpy", "--gui"], name="OptiCalib", icon="/i.png")
        lines = entry.splitlines()
        assert lines[0] == "[Desktop Entry]"
        assert "Exec=/env/bin/calpy --gui" in lines
        assert "Name=OptiCalib" in lines and "Icon=/i.png" in lines and "Terminal=false" in lines

    def test_existing_config(self, experiment) -> None:
        folder = experiment.parent.parent
        assert setup_calpy._existing_config(str(folder)) == str(experiment)
        assert setup_calpy._existing_config(str(experiment)) == str(experiment)
        with pytest.raises(FileNotFoundError):
            setup_calpy._existing_config(str(folder / "missing"))

    def test_launcher_names(self, experiment) -> None:
        assert setup_calpy._launcher_names(None) == ("OptiCalib", "OptiCalib")
        assert setup_calpy._launcher_names(str(experiment)) == ("OptiCalib (My Exp)", "OptiCalib-My_Exp")

    def test_install_and_uninstall_linux(self, linux_home, experiment) -> None:
        created = setup_calpy.install_launcher(str(experiment))
        menu = linux_home / ".local" / "share" / "applications" / "OptiCalib-My_Exp.desktop"
        desktop = linux_home / "Desktop" / "OptiCalib-My_Exp.desktop"
        assert created == [str(menu), str(desktop)]
        text = menu.read_text()
        assert "Name=OptiCalib (My Exp)" in text
        assert f'--gui -f "{experiment}"' in text
        assert os.access(desktop, os.X_OK)
        assert setup_calpy.uninstall_launcher(str(experiment)) == created
        assert not menu.exists() and not desktop.exists()

    def test_windows_shortcuts(self, monkeypatch, experiment) -> None:
        monkeypatch.setattr(setup_calpy.sys, "platform", "win32")
        monkeypatch.setattr(setup_calpy, "_opticalib_exe", lambda: r"C:\env\Scripts\OptiCalib.exe")
        scripts = []
        monkeypatch.setattr(setup_calpy, "_run_powershell", lambda script: scripts.append(script) or ["ok.lnk"])
        assert setup_calpy.install_launcher(str(experiment)) == ["ok.lnk"]
        script = scripts[-1]
        assert "CreateShortcut" in script and "'OptiCalib (My Exp).lnk'" in script
        assert r"$s.TargetPath = 'C:\env\Scripts\OptiCalib.exe'" in script
        assert f'$s.Arguments = \'-f "{experiment}"\'' in script
        assert "GetFolderPath('Desktop')" in script and "GetFolderPath('Programs')" in script
        setup_calpy.uninstall_launcher(str(experiment))
        assert "Remove-Item" in scripts[-1]

    def test_windows_without_exe(self, monkeypatch) -> None:
        monkeypatch.setattr(setup_calpy.sys, "platform", "win32")
        monkeypatch.setattr(setup_calpy, "_opticalib_exe", lambda: None)
        with pytest.raises(RuntimeError, match="OptiCalib.exe not found"):
            setup_calpy.install_launcher()

    def test_unsupported_platform(self, monkeypatch) -> None:
        monkeypatch.setattr(setup_calpy.sys, "platform", "darwin")
        with pytest.raises(RuntimeError, match="not supported"):
            setup_calpy.install_launcher()

    def test_powershell_quoting(self) -> None:
        assert setup_calpy._powershell_quote("it's") == "'it''s'"

    def test_cli(self, monkeypatch, experiment, capsys) -> None:
        calls = []
        monkeypatch.setattr(setup_calpy, "install_launcher", lambda config=None: calls.append(config) or ["/x.desktop"])
        monkeypatch.setattr(setup_calpy.sys, "argv", ["calpy", "--install-launcher", "-f", str(experiment.parent.parent)])
        setup_calpy.main()
        assert calls == [str(experiment)]
        assert "/x.desktop" in capsys.readouterr().out

    def test_gui_main(self, monkeypatch, experiment) -> None:
        launched = []
        monkeypatch.setattr(setup_calpy, "_launch_gui", lambda config_path=None, report=print: launched.append(config_path))
        monkeypatch.setattr(setup_calpy.sys, "argv", ["OptiCalib", "-f", str(experiment.parent.parent)])
        setup_calpy.gui_main()
        assert launched == [str(experiment)]
        assert os.environ["AOCONF"] == str(experiment)

    def test_packaging(self) -> None:
        root = Path(setup_calpy.__file__).resolve().parent
        setup_text = (root / "setup.py").read_text()
        assert '"OptiCalib=setup_calpy:gui_main"' in setup_text and '"gui_scripts"' in setup_text
        assert '"gui/resources/*"' in setup_text
        for ext in ("svg", "png", "ico"):
            assert (root / "opticalib" / "gui" / "resources" / f"opticalib.{ext}").is_file()
