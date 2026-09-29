# Configuration file for the Sphinx documentation builder.
#
# Full list of options: https://www.sphinx-doc.org/en/master/usage/configuration.html
from __future__ import annotations

import os
import sys
import warnings
from datetime import datetime

# astropy >= 8 deprecates ``astropy.samp`` in favour of ``pyvo.samp`` and warns
# about it at import time.  opticalib never touches SAMP, but the warning is
# emitted while ``import opticalib`` runs -- i.e. during this configuration file
# -- and Sphinx routes it to its own warning handler, where it counts against
# the build.
#
# NOTE: ``suppress_warnings = ["astropy.samp"]`` does NOT work here.  Sphinx's
# ``WarningSuppressor`` handler filter runs *before* ``WarningLogRecordTranslator``
# derives ``record.type``/``record.subtype``, so records originating from a
# third-party logger still have an empty type when suppression is evaluated.
# Filtering the Python warning at the source is the only reliable fix.
warnings.filterwarnings(
    "ignore",
    message=r".*astropy\.samp was deprecated.*",
)

# matplotlib's font cache emits ``findfont: Failed to find font weight bold, now
# using 400`` on headless builders.  It is harmless, but Sphinx attaches its
# warning handler to the root logger, so *any* third-party WARNING-level record
# counts against ``app._warncount`` -- and therefore against
# ``fail_on_warning`` on Read the Docs -- even though it is not a documentation
# problem.  Raising the level on that one logger keeps the build honest:
# ``sphinx-build -W`` then fails only on real doc regressions.
import logging  # noqa: E402  (must stay after the warnings filter above)

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# -- Path setup --------------------------------------------------------------
# Project root, so that ``import opticalib`` resolves during autodoc.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _bootstrap_aoconf() -> str:
    """Point ``AOCONF`` at a throwaway experiment before opticalib is imported.

    ``opticalib.core.root`` reads and *opens* the configuration file at import
    time, and creates the whole data folder tree as a side effect.  Without
    this step a documentation build would either fail outright (the shipped
    template is gitignored, so it is missing from a fresh clone) or silently
    create ``~/.opticalib`` on the build machine.

    Returns the path of the configuration file that will be used.
    """
    import shutil
    import tempfile

    existing = os.environ.get("AOCONF")
    if existing and os.path.isfile(existing):
        return existing

    stub = os.path.join(os.path.dirname(__file__), "_stubs", "configuration.yaml")
    template = os.path.join(
        ROOT, "opticalib", "core", "_configurations", "configuration.yaml"
    )
    src = stub if os.path.isfile(stub) else template

    workdir = os.path.join(tempfile.gettempdir(), f"opticalib-docs-{os.getpid()}")
    os.makedirs(workdir, exist_ok=True)
    runtime_conf = os.path.join(workdir, "configuration.yaml")

    if os.path.isfile(src):
        shutil.copyfile(src, runtime_conf)
    else:  # pragma: no cover - neither stub nor template available
        with open(runtime_conf, "w", encoding="utf-8") as fh:
            fh.write("SYSTEM:\n  data_path: ''\n")

    # Redirect data_path into the scratch directory so that create_folder_tree()
    # does not touch the user's home directory.
    try:
        from ruamel.yaml import YAML

        _yml = YAML()
        _yml.preserve_quotes = True
        with open(runtime_conf, encoding="utf-8") as fh:
            data = _yml.load(fh)
        data.setdefault("SYSTEM", {})["data_path"] = workdir
        with open(runtime_conf, "w", encoding="utf-8") as fh:
            _yml.dump(data, fh)
    except Exception:  # pragma: no cover - keep the build going
        pass

    os.environ["AOCONF"] = runtime_conf
    return runtime_conf


_AOCONF = _bootstrap_aoconf()

# -- Project information -----------------------------------------------------
project = "OptiCalib"
author = "P. Ferraiuolo, M. Xompero, R. Briguglio, A. Puglisi"
current_year = datetime.now().year
copyright = f"2024-{current_year}, {author}"  # noqa: A001 - Sphinx reserved name

try:
    from opticalib.__version__ import __version__ as release
except Exception:  # pragma: no cover - fallback when package is not importable
    release = "0.0.0"
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
extensions = [
    # --- Sphinx core ---
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.githubpages",
    "sphinx.ext.autosectionlabel",
    # --- Third-party ---
    "myst_parser",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_sitemap",
    "sphinxext.opengraph",
    "notfound.extension",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
source_encoding = "utf-8"
master_doc = "index"

exclude_patterns = [
    "_build",
    # NOTE: do NOT exclude "generated".  If autosummary is ever used with a
    # ``:toctree:`` option it writes its stubs there, and Sphinx must be able to
    # read them back; excluding the directory makes every API page silently
    # vanish ("stub file not found").  The curated reference in docs/reference/
    # uses explicit ``automodule`` directives and generates no stubs at all, so
    # the directory simply does not exist any more.
    "_stubs",           # docs-build configuration stub, not documentation
    "tools",            # maintenance scripts, not documentation
    "_prolog.rst",      # substitution definitions, injected via rst_prolog
    "requirements.txt",
    "Makefile",
    "make.bat",
    "*.mmd",
]

# Substitutions (|version|, |tn|, ...) available in every RST document.
_prolog_path = os.path.join(os.path.dirname(__file__), "_prolog.rst")
if os.path.isfile(_prolog_path):
    with open(_prolog_path, encoding="utf-8") as _fh:
        rst_prolog = _fh.read()

# -- Options for autodoc -----------------------------------------------------
# Deliberately conservative: document the *public* API only, grouped by kind.
# ``undoc-members``, ``private-members`` and ``inherited-members`` are
# intentionally NOT enabled -- they were the main source of noise in earlier
# builds, where every page was flooded with ``_private``, dunder and ABC
# members inherited from ``object``.
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "special-members": "__init__",
    "exclude-members": "__weakref__,__dict__,__module__,__qualname__,__slots__",
}
autodoc_member_order = "groupwise"
autodoc_class_signature = "separated"
autodoc_preserve_defaults = True
autodoc_inherit_docstrings = True

autosummary_generate = True
autosummary_generate_overwrite = True
autosummary_imported_members = False

# --- napoleon (Google + NumPy style docstrings) ---
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = False
napoleon_use_keyword = True
napoleon_attr_annotations = True
# Kept OFF deliberately: napoleon_preprocess_types rewrites ordinary prose in
# docstrings ("How", "Additional", "but", "number", ...) into :class: roles,
# which under nitpicky mode produces hundreds of bogus unresolved references.
napoleon_preprocess_types = False
napoleon_custom_sections = [
    ("Contents", "notes"),
    ("How to Use", "examples"),
    ("Other Parameters", "params_style"),
]

# --- sphinx-autodoc-typehints ---
# The package leans on the ``opticalib.core._types`` aliases and on
# ``from __future__ import annotations``, so hints need a little help.
autodoc_typehints = "signature"
autodoc_type_aliases = {
    "ImageData": "numpy.ma.MaskedArray",
    "CubeData": "numpy.ma.MaskedArray",
    "MaskData": "numpy.ma.MaskedArray",
    "MatrixLike": "numpy.typing.ArrayLike",
    "Header": "astropy.io.fits.Header",
    # Forward reference used by the ``Reconstructor`` alias in
    # ``opticalib/core/_types.py``; without this, sphinx-autodoc-typehints emits
    # an unqualified :class:`ComputeReconstructor` that cannot be resolved.
    "ComputeReconstructor": "opticalib.ground.reconstructor.ComputeReconstructor",
    "Reconstructor": "opticalib.ground.reconstructor.ComputeReconstructor",
    "xp": "numpy",
}
always_use_bars_union = True
# "comma"/"braces" inject the parameter default into the *type* field
# ("bool, default: False"), which the python domain then tries to resolve as a
# comma-separated list of types -- producing bogus ``py:class False`` targets
# under nitpicky mode.  Defaults are already visible in the signature because
# ``autodoc_typehints = "signature"``, so nothing is lost by leaving this off.
# Valid values are None, "braces", "comma" and "braces-after".
typehints_defaults = None
typehints_document_rtype = False
simplify_optional_union_types = True

# -- Mocking hardware / GPU-only third-party dependencies ---------------------
# These *third-party* packages require vendor SDKs, a CUDA runtime or a Qt
# display, none of which exist in a documentation build environment.
# ``autodoc_mock_imports`` replaces them with inert stubs so that
# ``import opticalib`` succeeds and autodoc can still introspect the real
# signatures of our own classes.
#
# NOTE: never mock first-party ``opticalib.*`` modules here.  Doing so hides
# real API pages instead of fixing them (this was a bug in earlier configs).
autodoc_mock_imports = [
    "asdk",         # Alpao deformable mirror SDK (vendor, not on PyPI)
    "cupy",         # CUDA arrays, pulled in by xupy
    "Microgate",    # Microgate controller SDK (vendor)
    "pipython",     # Physik Instrumente motor SDK
    "plico_motor",  # Arcetri motor server client
    "PyQt5",        # GUI only (opticalib.gui)
    "Pyro4",        # SPLATT DM remote objects
    "qtconsole",    # GUI only (opticalib.gui)
    "torch",        # heavy CUDA dependency of xupy
    "vmbpy",        # Allied Vision Vimba SDK bindings
    "xupy",         # CPU/GPU numpy duality layer
]
# ``arte`` is pure Python and installable, so it is installed rather than
# mocked -- see docs/requirements.txt.

# -- Options for intersphinx -------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "astropy": ("https://docs.astropy.org/en/stable", None),
    "skimage": ("https://scikit-image.org/docs/stable", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "h5py": ("https://docs.h5py.org/en/stable", None),
}
intersphinx_timeout = 15

# -- Strict reference checking ----------------------------------------------
# ``nitpicky`` turns every unresolved cross-reference into a warning so that
# dangling :class:`/:func:` targets cannot silently rot.  Known-unresolvable
# targets are listed in docs/nitpick_ignore.txt.
nitpicky = os.environ.get("OPTICALIB_DOCS_NITPICK", "1") == "1"
_ignore_path = os.path.join(os.path.dirname(__file__), "nitpick_ignore.txt")
if os.path.isfile(_ignore_path):
    nitpick_ignore: list[tuple[str, str]] = []
    nitpick_ignore_regex: list[tuple[str, str]] = []
    with open(_ignore_path, encoding="utf-8") as _fh:
        for _line in _fh:
            _line = _line.strip()
            if not _line or _line.startswith("#"):
                continue
            if _line.startswith("re:"):
                _kind, _target = _line[3:].split(None, 1)
                nitpick_ignore_regex.append((_kind.strip(), _target.strip()))
            else:
                _kind, _target = _line.split(None, 1)
                nitpick_ignore.append((_kind.strip(), _target.strip()))
del _ignore_path

suppress_warnings = [
    "myst.header",           # MyST documents manage their own heading levels
    "toc.duplicate_label",   # curated reference pages re-export symbols
    # ``sphinx.ext.githubpages`` writes ``.nojekyll`` and ``CNAME`` into the
    # output directory of *every* builder, and the epub builder then warns that
    # it cannot assign them a mimetype.  It also warns about the ``.doctrees``
    # directory when the doctree path lives inside the output tree.  Neither is
    # a documentation defect, but both are fatal under ``fail_on_warning`` --
    # which Read the Docs applies to the pdf and epub formats as well.
    "epub.unknown_project_files",
    # ``autosectionlabel`` labels every section of every document.  Module
    # docstrings rendered by autodoc carry their own headings ("Author(s)",
    # "Cameras", "Slaving", ...), which collide with the curated section
    # headings of the page that embeds them.  Nothing in these docs uses :ref:
    # against section labels (only genindex/modindex/search), so the collisions
    # are pure noise.
    "autosectionlabel.*",
    # NOTE: do NOT suppress the generic "autodoc" type.  It hides real failures
    # such as a module that autodoc could not import, which then silently
    # produces an empty reference page.
]

# -- Options for autosectionlabel --------------------------------------------
autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 3
del _prolog_path

# -- Options for MyST ---------------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "attrs_inline",
    "attrs_block",
    "substitution",
    "linkify",
    "tasklist",
    "smartquotes",
    "strikethrough",
]
myst_heading_anchors = 4
myst_ref_domains = ["py", "std", "doc"]
myst_footnote_transition = False

# Markdown equivalent of the substitutions in docs/_prolog.rst.
# Usable in MyST documents as {{ opt }}, {{ calpy }}, {{ tn }}, ...
myst_substitutions = {
    "version": version,
    "release": release,
    "opt": "OptiCalib",
    "calpy": "`calpy`",
    "tn": "`tn`",
    "yaml": "`configuration.yaml`",
    "ao": "INAF – Osservatorio Astrofisico di Arcetri",
    "AOGroup": "[Adaptive Optics Group](https://ao.arcetri.inaf.it/)",
    "GitHub": "[GitHub](https://github.com/ArcetriAdaptiveOptics/opticalib)",
    "RTD": "[Read the Docs](https://opticalib.readthedocs.io/)",
}

# -- Options for copybutton ---------------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = True
copybutton_line_continuation_character = "\\"
copybutton_selector = "div:not(.no-copybutton) > div.highlight > pre"

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_title = f"{project} {release}"
html_short_title = project
html_last_updated_fmt = "%d %b %Y"
html_show_sourcelink = False
html_show_copyright = True
html_show_sphinx = False
html_baseurl = "https://opticalib.readthedocs.io/en/latest/"
html_copy_source = False
html_scaled_image_link = False

_logo = os.path.join(os.path.dirname(__file__), "_static", "img", "logo.svg")
if os.path.isfile(_logo):
    html_logo = "_static/img/logo.svg"
_favicon = os.path.join(os.path.dirname(__file__), "_static", "img", "favicon.ico")
if os.path.isfile(_favicon):
    html_favicon = "_static/img/favicon.ico"
del _logo, _favicon

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#1a6db5",
        "color-brand-content": "#1a6db5",
        "color-api-background": "#f4f6f9",
        "color-inline-code-background": "#faeef0",
        "color-api-name": "#1a6db5",
        "color-api-pre-name": "#1a6db5",
        "color-link": "#1a6db5",
        "color-link--hover": "#0d4f85",
        "color-sidebar-background": "#fafbfc",
        "color-sidebar-brand-text": "#1a2b3c",
        "color-sidebar-caption-text": "#5a6b7c",
        "color-sidebar-link-text": "#3a4b5c",
        "color-sidebar-link-text--top-level": "#1a2b3c",
        "color-foreground-secondary": "#5a6b7c",
    },
    "dark_css_variables": {
        "color-brand-primary": "#5ba4e0",
        "color-brand-content": "#5ba4e0",
        "color-api-background": "#1b2128",
        "color-inline-code-background": "#2d1e20",
        "color-api-name": "#5ba4e0",
        "color-api-pre-name": "#5ba4e0",
        "color-link": "#5ba4e0",
        "color-link--hover": "#7dbce8",
        "color-sidebar-background": "#15191e",
        "color-sidebar-brand-text": "#c8d6e5",
        "color-sidebar-caption-text": "#6b7d90",
        "color-sidebar-link-text": "#8b9db0",
        "color-sidebar-link-text--top-level": "#c8d6e5",
        "color-foreground-secondary": "#6b7d90",
    },
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "announcement": None,
    # NB: do not set ``top_of_page_buttons`` -- the installed Furo release
    # rejects custom values and warns on every page.  Furo derives the
    # source/edit links from ``source_repository`` / ``source_branch`` below.
    "source_repository": "https://github.com/ArcetriAdaptiveOptics/opticalib/",
    "source_branch": "main",
    "source_directory": "docs/",
}

html_context = {
    "display_github": True,
    "github_user": "ArcetriAdaptiveOptics",
    "github_repo": "opticalib",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# -- Options for OpenGraph / sitemap -----------------------------------------
ogp_site_url = html_baseurl
ogp_site_name = f"{project} documentation"
ogp_type = "website"
ogp_enable_meta_description = True

# -- Options for the custom 404 page -----------------------------------------
notfound_context = {
    "title": "Page not found",
    "body": (
        "<h1>Page not found</h1>"
        "<p>The page you requested does not exist in this version of the "
        "OptiCalib documentation.</p>"
        "<ul>"
        '<li>Use the search box in the sidebar.</li>'
        '<li>Return to the <a href="/en/latest/index.html">documentation '
        "home</a>.</li>"
        "<li>Check the version selector at the bottom-left of the page.</li>"
        "</ul>"
    ),
}
notfound_urls_prefix = "/en/latest/"

# -- Options for TODOs -------------------------------------------------------
# Off by default; set OPTICALIB_DOCS_TODOS=1 for a local "draft" build.
todo_include_todos = os.environ.get("OPTICALIB_DOCS_TODOS", "0") == "1"
todo_emit_warnings = False

# -- Options for doctest -----------------------------------------------------
doctest_test_doctest_blocks = ""

# -- Options for LaTeX / PDF output ------------------------------------------
latex_elements = {
    "papersize": "a4paper",
    "pointsize": "11pt",
    "preamble": (
        r"\usepackage{fvextra}"
        r"\DefineVerbatimEnvironment{Highlighting}{Verbatim}"
        r"{breaklines,breakanywhere,commandchars=\\\{\}}"
    ),
    "figure_align": "htbp",
}
latex_documents = [
    (master_doc, "opticalib.tex", "OptiCalib Documentation", author, "manual"),
]

# -- Options for EPUB output --------------------------------------------------
epub_title = project
epub_author = author
epub_publisher = "INAF - Osservatorio Astrofisico di Arcetri"
epub_copyright = copyright
epub_exclude_files = ["search.html"]

