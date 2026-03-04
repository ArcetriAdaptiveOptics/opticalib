"""Tests for setup_calpy module.

Tests ensure the calpy entry point can locate initCalpy.py bootstrap script
both in development (source checkout) and in installed wheel packages.
"""

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
        package_data declaration for _init_script/initCalpy.py.
        """
        setup_path = Path(__file__).resolve().parents[1] / "setup.py"
        setup_source = setup_path.read_text(encoding="utf-8")
        assert "_init_script/initCalpy.py" in setup_source, (
            "setup.py does not declare _init_script/initCalpy.py in package_data. "
            "This will break calpy in installed wheels."
        )

    def test_manifest_includes_init_script(self) -> None:
        """Ensure MANIFEST.in includes packaged init script for sdists.
        
        Source distributions must include the packaged init script resource.
        """
        manifest_path = Path(__file__).resolve().parents[1] / "MANIFEST.in"
        manifest_source = manifest_path.read_text(encoding="utf-8")
        assert "_init_script/initCalpy.py" in manifest_source, (
            "MANIFEST.in does not declare _init_script/initCalpy.py. "
            "This will break calpy in source distributions."
        )
