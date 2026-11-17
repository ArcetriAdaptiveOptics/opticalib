# Building the documentation

The docs are built with Sphinx and published on {{ RTD }}. This page records
the non-obvious constraints of the build — most of them exist because
OptiCalib has hardware and GPU dependencies that cannot be installed in a
documentation environment.

## Building locally

```{code-block} bash
python -m venv /tmp/docenv && /tmp/docenv/bin/pip install -r docs/requirements.txt
cd docs && make html
```

Nitpicky mode is on by default; disable it temporarily with
`OPTICALIB_DOCS_NITPICK=0 make html` while drafting.

## The `AOCONF` bootstrap

{mod}`opticalib.core.root` reads its configuration **at import time** and calls
{func}`~opticalib.core.root.create_folder_tree`. Two consequences:

1. `import opticalib` fails if no configuration file can be found.
2. The shipped template `opticalib/core/_configurations/configuration.yaml` is
   **gitignored**, so a fresh clone has no configuration at all.

`docs/conf.py` therefore bootstraps a temporary environment before importing
the package: it points `AOCONF` at `docs/_stubs/configuration.yaml`, a
build-safe stub whose `data_path` is redirected into the Sphinx output
directory so the build never writes to a user's home folder.

```{warning}
If you change `docs/_stubs/configuration.yaml`, keep `data_path` empty or
pointing inside `_build`. A real path will cause the docs build to create
folders on the machine that builds them.
```

## What is mocked, and what is not

`autodoc_mock_imports` in `docs/conf.py` lists **third-party** packages that
need a vendor SDK, a CUDA runtime or a display:

`asdk`, `cupy`, `Microgate`, `pipython`, `plico_motor`, `PyQt5`, `Pyro4`,
`qtconsole`, `torch`, `vmbpy`, `xupy`

```{important}
Never add a first-party `opticalib.*` module to this list. Mocking our own
package hides real API pages instead of fixing them, and produces reference
sections that look complete but document nothing. This was a bug in earlier
configurations.
```

Pure-Python third-party dependencies (`arte`, `numpy`, `astropy`, ...) are
installed normally via `docs/requirements.txt`.

## Structure of the reference

`docs/api.rst` is a landing page; the content lives in `docs/reference/*.rst`,
one page per functional area. Each page uses explicit `.. automodule::`
directives for the modules that **define** its symbols.

This matters: documenting a package `__init__` with `:recursive:` documents both
the definition site and every re-export, which makes Sphinx report
*duplicate object description* for dozens of symbols. The curated pages avoid
that by documenting each symbol exactly once, and listing re-exports in a
"backward-compatible aliases" table instead.

`autosummary` is used only for navigation tables (no `:toctree:`), so no stub
files are generated and nothing needs to be committed under `docs/generated/`.

## Nitpicky mode and `nitpick_ignore.txt`

`nitpicky = True` turns every unresolved cross-reference into a warning. Genuinely
unresolvable targets — things from mocked packages, or `typing` constructs with
no intersphinx entry — go in `docs/nitpick_ignore.txt`.

````{important}
Sphinx matches `nitpick_ignore_regex` entries with {func}`re.fullmatch`, not
{func}`re.search`. A pattern like `^py:class opticalib\._private\..*` never
matches, because the leading `^` is redundant and the pattern must consume the
**entire** target string. Write patterns without anchors and let them match
end to end:

```{code-block} text
# correct
re:py:class opticalib\.core\._types\..*

# silently does nothing
re:py:class ^opticalib\.core\._types\..*
```
````

Lines without the `re:` prefix are treated as exact `(domain, target)` pairs.

## Things that have bitten us before

```{list-table}
:header-rows: 1
:widths: 34 66

* - Symptom
  - Cause and fix
* - Every API page silently empty, `autosummary` "stub file not found"
  - `docs/generated` was listed in `exclude_patterns`. Sphinx could not read
    back the stubs it had just written. Do not exclude it.
* - Hundreds of unresolved refs to ordinary English words
  (`How`, `Additional`, `number`, ...)
  - `napoleon_preprocess_types = True` rewrites prose into `:class:` roles.
    It is disabled in `docs/conf.py`.
* - `duplicate object description` for every method of a class
  - The class docstring had a numpydoc **`Methods`** section. Napoleon's
    `_parse_methods_section` emits a literal `.. method::` directive for each
    entry, so every method is described twice -- once by autodoc, once by the
    docstring. Delete the `Methods` section; autodoc already lists the real
    methods with real signatures.
* - `py:class reference target not found: N` / `2` / `False` / `Header`
  - A numpydoc *type* field containing prose that Sphinx splits on commas or
    pipes: `points : (N,2) ndarray`, `no_class : bool, False`,
    `cube_header : dict | Header`. Write real, resolvable type names
    (`numpy.ndarray`, `astropy.io.fits.Header`) and move the prose into the
    description.
* - `Block quote ends without a blank line; unexpected unindent`
  - A bullet list that starts on the line immediately after the parameter
    description. Insert a blank line before the first `-`.
* - `Inline strong start-string without end-string`
  - `**kwargs` written into a docstring section that napoleon did *not*
    recognise, so it was passed through as literal RST where `**` opens a
    `strong` node. The usual cause is a mistyped section heading -- see the
    next row.
* - A whole `Parameters` block renders as raw text
  - The heading was written `Parameters:` with a trailing colon. Napoleon
    matches section names exactly, so `Parameters:` is not a section at all.
* - An entire module's page comes out empty, with no warning
  - Every documented object in it lacked a docstring. autodoc skips
    undocumented members unless `:undoc-members:` is given, and if the module
    itself has no docstring either, `automodule` emits nothing at all. Add the
    docstrings -- do not paper over it with `:undoc-members:`.
* - `typehints_defaults needs to be one of {None, 'braces', 'comma', ...}`
  - The value is the Python object `None`, not the string `"none"`.
* - A third-party warning keeps counting against the build even though it is
  listed in `suppress_warnings`
  - `suppress_warnings` cannot suppress records that originate from another
    package's logger. Sphinx's `WarningSuppressor` handler filter runs *before*
    `WarningLogRecordTranslator` derives `record.type` / `record.subtype`, so
    the type is still empty when suppression is evaluated. Filter the Python
    warning at the source in `conf.py` instead — see the
    `warnings.filterwarnings` call at the top of the file.
* - Furo build fails on an unknown theme option
  - Unsupported `html_theme_options` keys are hard errors. Check the Furo
    release notes before adding one.
* - `opticalib.core.root` import error on a fresh clone
  - Missing `AOCONF` and the gitignored configuration template. See the
    bootstrap section above.
```

```{important}
Do not add `"autodoc"` to `suppress_warnings`. It hides genuine failures --
including modules that autodoc could not import -- and leaves reference pages
that look fine in the log but are empty in the browser.
```

## Publishing

`.readthedocs.yaml` drives the RTD build. It installs the project itself via a
`post_install` step, because autodoc needs the *real* package importable to
read signatures — installing only `docs/requirements.txt` is not enough.
