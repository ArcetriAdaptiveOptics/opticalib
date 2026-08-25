"""Tests for setup_calpy module.

Tests ensure the calpy entry point can locate initCalpy.py bootstrap script
both in development (source checkout) and in installed wheel packages.
"""

import os
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
