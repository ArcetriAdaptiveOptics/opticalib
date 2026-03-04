"""Tests for setup_calpy module."""

from pathlib import Path

import setup_calpy


class TestResolveInitFile:
    """Test initCalpy script resolution logic."""

    def test_resolve_prefers_packaged_script(self, monkeypatch, temp_dir):
        """Return packaged script path when both candidates are available."""
        base_dir = Path(temp_dir)
        packaged = base_dir / "opticalib" / "_init_script" / "initCalpy.py"
        local_dev = base_dir / "__init_script__" / "initCalpy.py"

        packaged.parent.mkdir(parents=True, exist_ok=True)
        packaged.write_text("# packaged")
        local_dev.parent.mkdir(parents=True, exist_ok=True)
        local_dev.write_text("# local")

        monkeypatch.setattr(setup_calpy, "__file__", str(base_dir / "setup_calpy.py"))

        resolved = setup_calpy._resolve_init_file()
        assert resolved == str(packaged)

    def test_resolve_falls_back_to_local_dev_script(self, monkeypatch, temp_dir):
        """Return local dev script path when packaged script is missing."""
        base_dir = Path(temp_dir)
        local_dev = base_dir / "__init_script__" / "initCalpy.py"

        local_dev.parent.mkdir(parents=True, exist_ok=True)
        local_dev.write_text("# local")

        monkeypatch.setattr(setup_calpy, "__file__", str(base_dir / "setup_calpy.py"))

        resolved = setup_calpy._resolve_init_file()
        assert resolved == str(local_dev)

    def test_resolve_returns_none_when_missing(self, monkeypatch, temp_dir):
        """Return None when no init script candidate exists."""
        base_dir = Path(temp_dir)
        monkeypatch.setattr(setup_calpy, "__file__", str(base_dir / "setup_calpy.py"))

        resolved = setup_calpy._resolve_init_file()
        assert resolved is None


def test_setup_declares_packaged_init_script():
    """Keep setup metadata aligned with calpy init script packaging."""
    setup_path = Path(__file__).resolve().parents[1] / "setup.py"
    setup_source = setup_path.read_text(encoding="utf-8")

    assert "_init_script/initCalpy.py" in setup_source
