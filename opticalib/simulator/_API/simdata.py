"""Utility functions to resolve and fetch simulator static data files."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from urllib import error as _urlerr
from urllib import request as _urlreq

from tqdm import tqdm as _tqdm

from ...core import root as _root

_DEFAULT_BASE_URL = (
    "https://github.com/ArcetriAdaptiveOptics/opticalib/"
    "releases/download/v1.3.1-simdata"
)


_SIMDATA_FILES: dict[str, dict[str, str]] = {
    "m4_data.h5": {
        "sha256": (
            "9ec137128f32a637d9a894fe2a3cecdfa775675a3b44e982"
            "51a0a8587e8184b0"
        )
    },
    "dp_cmdmat.fits": {
        "sha256": (
            "0b18e52ab70af9c44bc36f929ed3cdda23872a0b3d2268df"
            "11fb42be0c59db88"
        )
    },
    "dp_coords.fits": {
        "sha256": (
            "223e65bd07b4e9f8a18126eaf8b2faab2f672f677dc014334"
            "67c67f2aed99931"
        )
    },
    "dp_ffwd.fits": {
        "sha256": (
            "485c6b9dc03a3666e080e819328e004b53ebdb4822e5d7541"
            "d75cbf3799319e6"
        )
    },
    "ptl_init.fits": {
        "sha256": (
            "4e66c1da156221717ab979caf154a88cd2fe032b5e26a23ed"
            "700aea7c09198e5"
        )
    },
}


def available_simdata_files() -> list[str]:
    """
    Return the list of known simulator data file names.

    Returns
    -------
    list[str]
        Known simulator files that can be resolved from package or cache.
    """
    return sorted(_SIMDATA_FILES.keys())


def get_simdata_file(filename: str, auto_download: bool = True) -> str:
    """
    Resolve the full path of a simulator data file.

    Resolution order is:
    1) local cache folder under the user opticalib data path
    2) optional HTTP download into cache

    Parameters
    ----------
    filename : str
        Name of the file in the simulator SimData set.
    auto_download : bool, optional
        If True and file is missing locally, download from configured URL.

    Returns
    -------
    str
        Absolute path to the resolved file.

    Raises
    ------
    FileNotFoundError
        If file is unavailable and automatic download is disabled or fails.
    """
    cache_file = _cache_simdata_dir() / filename
    if cache_file.exists():
        _validate_if_known(cache_file, filename)
        return str(cache_file)

    if auto_download:
        _download_to_cache(filename, cache_file)
        return str(cache_file)

    raise FileNotFoundError(_missing_file_message(filename))


def prefetch_simdata(
    filenames: list[str] | None = None,
    force_download: bool = False,
) -> list[str]:
    """
    Ensure one or more simulator data files are available in local cache.

    Parameters
    ----------
    filenames : list[str] | None, optional
        File names to fetch. If None, fetch all known files.
    force_download : bool, optional
        If True, always download to cache even when cache file already exists.

    Returns
    -------
    list[str]
        List of resolved absolute paths.
    """
    wanted = filenames or available_simdata_files()
    resolved: list[str] = []
    for name in wanted:
        cache_file = _cache_simdata_dir() / name
        if force_download:
            _download_to_cache(name, cache_file)
            resolved.append(str(cache_file))
            continue
        resolved.append(get_simdata_file(name, auto_download=True))
    return resolved


def _cache_simdata_dir() -> Path:
    cache_dir = Path(_root.CONFIGURATION_FOLDER) / "SimData"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _download_to_cache(filename: str, target: Path) -> None:
    url = _resolve_download_url(filename)
    tmp_target = target.with_suffix(target.suffix + ".part")

    try:
        with _urlreq.urlopen(url, timeout=120) as response:
            total = int(response.headers.get("Content-Length", 0)) or None
            chunk_size = 8 * 1024 * 1024
            with open(tmp_target, "wb") as fout, _tqdm(
                total=total,
                unit="B",
                ncols=90,
                unit_scale=True,
                unit_divisor=1024,
                desc=filename,
            ) as bar:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    fout.write(chunk)
                    bar.update(len(chunk))
    except (_urlerr.URLError, TimeoutError) as exc:
        if tmp_target.exists():
            tmp_target.unlink(missing_ok=True)
        raise FileNotFoundError(
            f"Cannot download simulator data file '{filename}' from '{url}': {exc}"
        ) from exc

    _validate_if_known(tmp_target, filename)
    os.replace(tmp_target, target)


def _resolve_download_url(filename: str) -> str:
    env_key = f"OPTICALIB_SIMDATA_URL_{filename.upper().replace('.', '_')}"
    specific_url = os.getenv(env_key)
    if specific_url:
        return specific_url

    base_url = os.getenv("OPTICALIB_SIMDATA_BASE_URL", _DEFAULT_BASE_URL)
    return f"{base_url.rstrip('/')}/{filename}"


def _validate_if_known(path: Path, filename: str) -> None:
    expected = _SIMDATA_FILES.get(filename, {}).get("sha256")
    if expected is None:
        return

    digest = hashlib.sha256()
    with open(path, "rb") as fin:
        for chunk in iter(lambda: fin.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise FileNotFoundError(
            f"File checksum mismatch for '{filename}'. "
            "Please verify the configured SimData URL."
        )


def _missing_file_message(filename: str) -> str:
    base = os.getenv("OPTICALIB_SIMDATA_BASE_URL", _DEFAULT_BASE_URL)
    return (
        f"Simulator data file '{filename}' not found. "
        "Set OPTICALIB_SIMDATA_BASE_URL to a valid host (or a specific "
        f"OPTICALIB_SIMDATA_URL_{filename.upper().replace('.', '_')}) and retry. "
        f"Current base URL: '{base}'."
    )
