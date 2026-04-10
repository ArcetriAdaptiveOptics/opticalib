"""
Tests for folder-reference propagation after the GUI / calpy initialisation
sequence.

Background
----------
``opticalib.core.root`` reads the ``AOCONF`` environment variable **once at
module-import time** and derives every data-folder path from it.  Several
other modules (``iff_processing``, ``iff_module``, ``osutils``, ``read_config``)
cache their own copies of these paths at their own import time.

The GUI's ``_run_init_script`` method (and the equivalent CLI path) must:

1. Set ``AOCONF`` to the chosen config file.
2. Reload ``opticalib.core.root`` so its module-level globals are refreshed.
3. Rebind ``opticalib.folders`` to the new :class:`~opticalib.core.root._folds`
   instance that the reload created.
4. Call each dependent module's ``_update_imports()`` to propagate the new
   paths to their own cached globals.

After this sequence, every module that ``initCalpy.py`` imports (``dmutils``,
``iff_processing``, ``iff_module``, ``osutils``, ``read_config``) must expose
folder paths that are consistent with the chosen configuration file, so that
data is saved in the correct location (e.g. via
``dmutils.iff_module.iffDataAcquisition``).

These tests reproduce that sequence programmatically and assert consistency
without requiring a running Qt application or the ``xupy`` package.
"""

import importlib
import os
import shutil

import pytest
from ruamel.yaml import YAML as _YAML


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_yaml_parser = _YAML()
_yaml_parser.preserve_quotes = True


def _make_config(temp_dir: str, data_path: str) -> str:
    """
    Copy the package template configuration file, patch ``data_path``, and
    return the path to the new file.

    Parameters
    ----------
    temp_dir : str
        Directory in which to create the temporary configuration file.
    data_path : str
        Value to write into ``SYSTEM.data_path``.

    Returns
    -------
    str
        Absolute path to the temporary configuration file.
    """
    import opticalib.core.root as _root_mod

    config_path = os.path.join(temp_dir, "configuration.yaml")
    shutil.copy(_root_mod.TEMPLATE_CONF_FILE, config_path)
    with open(config_path, "r") as fh:
        config = _yaml_parser.load(fh)
    config["SYSTEM"]["data_path"] = data_path
    with open(config_path, "w") as fh:
        _yaml_parser.dump(config, fh)
    return config_path


def _run_init_sequence(config_path: str) -> None:
    """
    Execute the same path-propagation steps that the GUI's ``_run_init_script``
    performs, without starting a Qt application or running ``initCalpy.py``.

    Parameters
    ----------
    config_path : str
        Absolute path to the ``configuration.yaml`` file to activate.
    """
    import opticalib
    import opticalib.core.root as root_mod
    import opticalib.core.read_config as rc
    import opticalib.dmutils.iff_processing as ifp
    import opticalib.dmutils.iff_module as ifm
    import opticalib.ground.osutils as osu

    os.environ["AOCONF"] = config_path
    importlib.reload(root_mod)
    opticalib.folders = root_mod.folders
    rc._update_imports()
    osu._update_imports()
    ifp._update_imports()
    ifm._update_imports()


def _restore_init_sequence() -> None:
    """
    Undo the side-effects of :func:`_run_init_sequence` by re-running it with
    ``AOCONF`` cleared so that the template configuration is used again.

    This keeps the global module state clean between tests.
    """
    import opticalib
    import opticalib.core.root as root_mod
    import opticalib.core.read_config as rc
    import opticalib.dmutils.iff_processing as ifp
    import opticalib.dmutils.iff_module as ifm
    import opticalib.ground.osutils as osu

    os.environ.pop("AOCONF", None)
    importlib.reload(root_mod)
    opticalib.folders = root_mod.folders
    rc._update_imports()
    osu._update_imports()
    ifp._update_imports()
    ifm._update_imports()


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def init_with_config(tmp_path):
    """
    Set up a temporary configuration file, run the full init sequence, yield
    the expected paths, then restore module state.

    Yields
    ------
    tuple[str, str]
        ``(data_path, config_path)`` – the ``data_path`` embedded in the
        config and the path to the config file itself.
    """
    data_path = str(tmp_path / "TestData")
    config_path = _make_config(str(tmp_path), data_path)
    _run_init_sequence(config_path)
    try:
        yield data_path, config_path
    finally:
        _restore_init_sequence()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFolderPropagationAfterInit:
    """
    Verify that every module which ``initCalpy.py`` exposes (or which backs
    a user-facing function like ``iffDataAcquisition``) has folder references
    consistent with the chosen configuration file after the init sequence.
    """

    def test_opticalib_folders_base_data_path(self, init_with_config):
        """
        ``opticalib.folders.BASE_DATA_PATH`` must equal the ``data_path``
        written in the configuration file.
        """
        import opticalib

        data_path, _ = init_with_config
        assert opticalib.folders.BASE_DATA_PATH == data_path

    def test_opticalib_folders_configuration_file(self, init_with_config):
        """
        ``opticalib.folders.CONFIGURATION_FILE`` must match the path of the
        config file that was passed to the init sequence.
        """
        import opticalib

        _, config_path = init_with_config
        assert opticalib.folders.CONFIGURATION_FILE == config_path

    def test_opticalib_folders_opt_data_root(self, init_with_config):
        """
        ``opticalib.folders.OPT_DATA_ROOT_FOLDER`` must be rooted under the
        data path from the configuration file.
        """
        import opticalib

        data_path, _ = init_with_config
        assert opticalib.folders.OPT_DATA_ROOT_FOLDER.startswith(data_path)

    def test_opticalib_folders_iffunctions_root(self, init_with_config):
        """
        ``opticalib.folders.IFFUNCTIONS_ROOT_FOLDER`` must be rooted under
        the data path from the configuration file.
        """
        import opticalib

        data_path, _ = init_with_config
        assert opticalib.folders.IFFUNCTIONS_ROOT_FOLDER.startswith(data_path)

    # ------------------------------------------------------------------
    # read_config
    # ------------------------------------------------------------------

    def test_read_config_cfold(self, init_with_config):
        """
        ``read_config._cfold`` must equal
        ``opticalib.folders.CONFIGURATION_FOLDER`` after ``_update_imports``
        is called.
        """
        import opticalib
        import opticalib.core.read_config as rc

        assert rc._cfold == opticalib.folders.CONFIGURATION_FOLDER

    def test_read_config_cfile(self, init_with_config):
        """
        ``read_config._cfile`` must equal
        ``opticalib.folders.CONFIGURATION_FILE`` after ``_update_imports`` is
        called.
        """
        import opticalib
        import opticalib.core.read_config as rc

        assert rc._cfile == opticalib.folders.CONFIGURATION_FILE

    def test_read_config_iffold(self, init_with_config):
        """
        ``read_config._iffold`` must equal
        ``opticalib.folders.IFFUNCTIONS_ROOT_FOLDER`` after
        ``_update_imports`` is called.
        """
        import opticalib
        import opticalib.core.read_config as rc

        assert rc._iffold == opticalib.folders.IFFUNCTIONS_ROOT_FOLDER

    # ------------------------------------------------------------------
    # osutils
    # ------------------------------------------------------------------

    def test_osutils_optdata(self, init_with_config):
        """
        ``osutils._OPTDATA`` must equal
        ``opticalib.folders.OPT_DATA_ROOT_FOLDER`` after ``_update_imports``
        is called, so that ``findTracknum`` searches the correct tree.
        """
        import opticalib
        import opticalib.ground.osutils as osu

        assert osu._OPTDATA == opticalib.folders.OPT_DATA_ROOT_FOLDER

    # ------------------------------------------------------------------
    # iff_processing
    # ------------------------------------------------------------------

    def test_iff_processing_fn_iffunctions_root(self, init_with_config):
        """
        ``iff_processing._fn.IFFUNCTIONS_ROOT_FOLDER`` must equal
        ``opticalib.folders.IFFUNCTIONS_ROOT_FOLDER`` after
        ``_update_imports`` is called.
        """
        import opticalib
        import opticalib.dmutils.iff_processing as ifp

        assert (
            ifp._fn.IFFUNCTIONS_ROOT_FOLDER
            == opticalib.folders.IFFUNCTIONS_ROOT_FOLDER
        )

    def test_iff_processing_fn_intmat_root(self, init_with_config):
        """
        ``iff_processing._fn.INTMAT_ROOT_FOLDER`` must equal
        ``opticalib.folders.INTMAT_ROOT_FOLDER`` after ``_update_imports`` is
        called.
        """
        import opticalib
        import opticalib.dmutils.iff_processing as ifp

        assert ifp._fn.INTMAT_ROOT_FOLDER == opticalib.folders.INTMAT_ROOT_FOLDER

    def test_iff_processing_iffold(self, init_with_config):
        """
        ``iff_processing._ifFold`` must equal
        ``opticalib.folders.IFFUNCTIONS_ROOT_FOLDER`` after
        ``_update_imports`` is called.
        """
        import opticalib
        import opticalib.dmutils.iff_processing as ifp

        assert ifp._ifFold == opticalib.folders.IFFUNCTIONS_ROOT_FOLDER

    def test_iff_processing_intmatfold(self, init_with_config):
        """
        ``iff_processing._intMatFold`` must equal
        ``opticalib.folders.INTMAT_ROOT_FOLDER`` after ``_update_imports``
        is called.
        """
        import opticalib
        import opticalib.dmutils.iff_processing as ifp

        assert ifp._intMatFold == opticalib.folders.INTMAT_ROOT_FOLDER

    # ------------------------------------------------------------------
    # iff_module
    # ------------------------------------------------------------------

    def test_iff_module_fn_iffunctions_root(self, init_with_config):
        """
        ``iff_module._fn.IFFUNCTIONS_ROOT_FOLDER`` must equal
        ``opticalib.folders.IFFUNCTIONS_ROOT_FOLDER`` after
        ``_update_imports`` is called.

        This is the path that :func:`~opticalib.dmutils.iff_module.iffDataAcquisition`
        uses to write acquisition data, so it must be correct for data to be
        saved in the right location.
        """
        import opticalib
        import opticalib.dmutils.iff_module as ifm

        assert (
            ifm._fn.IFFUNCTIONS_ROOT_FOLDER
            == opticalib.folders.IFFUNCTIONS_ROOT_FOLDER
        )

    def test_iff_module_fn_base_data_path(self, init_with_config):
        """
        ``iff_module._fn.BASE_DATA_PATH`` must be rooted under the data path
        from the configuration file.
        """
        data_path, _ = init_with_config
        import opticalib.dmutils.iff_module as ifm

        assert ifm._fn.BASE_DATA_PATH == data_path

    # ------------------------------------------------------------------
    # Cross-module consistency
    # ------------------------------------------------------------------

    def test_all_iffunctions_paths_consistent(self, init_with_config):
        """
        All three module-level ``IFFUNCTIONS_ROOT_FOLDER`` references must
        be identical strings so that read and write operations across modules
        always target the same directory.
        """
        import opticalib
        import opticalib.core.read_config as rc
        import opticalib.dmutils.iff_module as ifm
        import opticalib.dmutils.iff_processing as ifp

        expected = opticalib.folders.IFFUNCTIONS_ROOT_FOLDER
        assert rc._iffold == expected, "read_config._iffold mismatch"
        assert ifp._ifFold == expected, "iff_processing._ifFold mismatch"
        assert (
            ifm._fn.IFFUNCTIONS_ROOT_FOLDER == expected
        ), "iff_module._fn.IFFUNCTIONS_ROOT_FOLDER mismatch"

    def test_all_intmat_paths_consistent(self, init_with_config):
        """
        Both ``INTMAT_ROOT_FOLDER`` references used by ``iff_processing``
        must be identical so that interaction-matrix cubes are saved and
        read from the same directory.
        """
        import opticalib
        import opticalib.dmutils.iff_processing as ifp

        expected = opticalib.folders.INTMAT_ROOT_FOLDER
        assert ifp._intMatFold == expected, "iff_processing._intMatFold mismatch"
        assert (
            ifp._fn.INTMAT_ROOT_FOLDER == expected
        ), "iff_processing._fn.INTMAT_ROOT_FOLDER mismatch"
