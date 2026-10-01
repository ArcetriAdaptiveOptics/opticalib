(user-guide-calpy)=

# The `calpy` experiment framework

`calpy` is the command-line entry point that ships with {{ opt }}. It does two
things:

1. It creates and maintains an **experiment** — a self-contained directory with a
   configuration file and the full data folder tree.
2. It drops you into an **IPython shell with {{ opt }} already wired up**, so
   common objects are one keystroke away instead of a dozen imports.

This page explains the experiment model, the shell namespace, and the standard
workflow. For the raw flag list see [](user-guide-calpy-cli).

## Why an experiment directory?

An adaptive-optics bench accumulates a lot of state: which mirror is installed,
where the interferometer writes its files, which influence-function captures
have been taken, and the calibration matrices derived from them. `calpy`
bundles all of that into one directory so that:

- a calibration campaign is **reproducible** — the exact
  {{ yaml }} that produced the data is archived inside the data tree;
- several campaigns can coexist without interfering, because each has its own
  root;
- scripts written inside the shell keep working later, because the data layout
  is fixed and discoverable through {{ opt }}'s `folders` mapping.

### The folder tree

Creating an experiment lays out:

```{code-block} text
alpao_experiment/
├── OPTData/
│   ├── Flattening/          # flat-command solutions and loops
│   ├── INTMatrices/         # interaction / control matrices
│   ├── ModalBases/          # Zernike, KL, RBF bases
│   ├── OPDImages/           # single processed phase maps
│   ├── OPDSeries/           # time-series measurements
│   ├── SPL/
│   │   └── Fringes/         # phasing interferograms
│   └── IFFunctions/
│       └── <tn>/            # one sub-folder per tracking number
├── Logging/                 # rotating text logs
└── SysConfig/
    └── configuration.yaml   # the experiment's configuration
```

Every acquisition is filed under a **tracking number** (`tn`), a timestamp
string such as `20260101_120000`. The `tn` is the join key between raw data,
processed data, logs and the configuration snapshot taken at acquisition time —
so `osu.find_tracknum`, `ifp.process(tn)` and `Flattening(tn=...)` all refer to
the same run.

(user-guide-calpy-cli)=

## Command-line usage

```{list-table}
:header-rows: 1
:widths: 40 60

* - Invocation
  - Effect
* - `calpy`
  - Shell using the default configuration at `~/.tmp_opticalib/SysConfig/configuration.yaml`
* - `calpy -f <path>`
  - Shell using the configuration at `<path>`
* - `calpy -f <path> --create`
  - Create the experiment at `<path>`, **then** enter the shell
* - `calpy --create <path>`
  - Create the experiment at `<path>` and **exit** (script-friendly)
* - `calpy -f <path> --gui`
  - Launch the Qt GUI for that experiment
* - `calpy --gui`
  - Launch the Qt GUI with the default configuration
```

`<path>` may be a directory or a direct path to a `.yaml` file. A relative path
is resolved against the current working directory and `~` is expanded. If the
directory does not exist it is created.

```{tip}
`--create` rewrites `SYSTEM.data_path` in the generated {{ yaml }} to the
directory you passed, so the new experiment is immediately self-consistent —
you do not have to edit the path by hand.
```

## The shell namespace

`calpy` executes the `opticalib/__init_script__/initCalpy.py` bootstrap at
startup. That module imports the library and binds the short aliases used
throughout this documentation:

```{list-table}
:header-rows: 1
:widths: 26 42 32

* - Name
  - Refers to
  - Used for
* - `opt`, `opticalib`
  - the `opticalib` package
  - device classes, top-level helpers
* - `np`
  - `numpy`
  - arrays
* - `xp`
  - `xupy`
  - CPU/GPU-transparent arrays
* - `dmutils`
  - {mod}`opticalib.dmutils`
  - calibration utilities
* - `procedures`
  - {mod}`opticalib.procedures`
  - stateful bench operations
* - `ifp`
  - {mod}`opticalib.dmutils.iff_processing`
  - influence-function reduction
* - `ifm`
  - {mod}`opticalib.procedures.iff`
  - influence-function acquisition
* - `folders`, `opaths`
  - `opticalib.folders`
  - absolute paths of the data tree
* - `osutils`, `osu`
  - {mod}`opticalib.ground.osutils`
  - tracking numbers, FITS/HDF5 I/O
* - `analyzer`, `az`
  - {mod}`opticalib.analyzer`
  - offline analysis
* - `modal_decomposer`, `zern`
  - {mod}`opticalib.ground.modal_decomposer`
  - Zernike / KL / RBF fitting
* - `roi`
  - {mod}`opticalib.ground.roi`
  - regions of interest
* - `simulator`, `sim`
  - {mod}`opticalib.simulator`
  - fake devices
* - `oplt`
  - {mod}`opticalib.visualization`
  - plotting helpers
* - `join`
  - `os.path.join`
  - path building
```

Matplotlib is star-imported (`from matplotlib.pyplot import *`) and interactive
mode is switched on, so `plot(...)`, `imshow(...)` and `show()` work directly
and figures update live.

```{warning}
The star-import means short Matplotlib names such as `plot`, `figure`, `title`
and `close` occupy the shell namespace. That is convenient interactively, but
in scripts prefer explicit imports so the origin of each name stays obvious.
```

## A complete calibration session

This is the canonical DM-calibration campaign, as you would type it in the
shell. The aliases (`opt`, `ifm`, `ifp`, `osu`, ...) come from the namespace
table above.

### 1. Create and open the experiment

```{code-block} bash
calpy -f ~/alpao_experiment --create
```

Edit `~/alpao_experiment/SysConfig/configuration.yaml` to describe your
hardware — see {doc}`/configuration`. Then reopen:

```{code-block} bash
calpy -f ~/alpao_experiment
```

### 2. Connect to the devices

Device constructors read their own entry from the `DEVICES` section, so on a
correctly configured bench they need no arguments:

```{code-block} python
interf = opt.PhaseCam()          # interferometer
dm     = opt.AlpaoDm()           # deformable mirror
print(dm.n_acts)
```

```{note}
Passing `ip=`, `port=` or `serial_number=` overrides the configuration for that
session only — useful when several devices of the same kind are declared.
```

### 3. Plan the influence-function capture

{class}`~opticalib.dmutils.iff_preparation.IFFCapturePreparation` turns the
`INFLUENCE.FUNCTIONS` configuration into the concrete command history that will
be sent to the mirror:

```{code-block} python
prep = dmutils.IFFCapturePreparation(dm)

# Build the timed command history from the configured modes/amplitudes
cmhist = prep.create_timed_cmd_history()

# What would be archived alongside the data
info = prep.get_info_to_save()
```

Arguments you pass override the configuration; omitted ones are taken from
{{ yaml }}. `create_cmd_matrix_history` and `create_aux_cmd_history` build the
other two blocks (trigger and registration) of the sequence.

### 4. Acquire

{func}`~opticalib.procedures.iff.iff_data_acquisition` runs the whole
push–pull loop and returns the tracking number:

```{code-block} python
tn = ifm.iff_data_acquisition(dm, interf)
tn
# '20260101_120000'
```

Everything the loop used — amplitudes, mode list, template, timing — is
snapshotted into the configuration copy stored with the data, so the capture is
self-describing.

### 5. Inspect what was captured

```{code-block} python
from opticalib.core.data_classes import IffData

iffdata = IffData(tn)
iffdata.modes_list         # which modes were pushed
iffdata.template           # push-pull sequence
iffdata.command_matrix     # commands actually sent
iffdata.iff                # the acquired influence functions
```

Load the acquired cube through the OS helpers, which resolve a `tn` to the
matching file list:

```{code-block} python
cube = osu.load_cube_from_filelist(tn)
cube.shape
```

### 6. Build the reconstructor

```{code-block} python
recon = opt.ground.reconstructor.ComputeReconstructor(tn=tn)
recon.run(sv_threshold=30)   # SVD-truncated reconstruction matrix
recon.IM                     # interaction matrix
recon.RM                     # reconstruction matrix
```

Pass `interactive=True` to `run` to pick the singular-value cutoff by hand with
the eigenvalue plot.

### 7. Flatten the mirror

{class}`~opticalib.dmutils.flattening.Flattening` is keyed by tracking number —
it loads the interaction matrix produced from that capture:

```{code-block} python
flat = dmutils.Flattening(tn, dm, interf)

flat.compute_rec_mat(threshold=30)
cmd  = flat.compute_flat_cmd(modes2flat=30)
flat.apply_flat_command()                 # send it and measure the result
flat.closed_loop_flattening(iterations=3) # iterate to convergence
```

The solved command and the measured images are archived as a
{class}`~opticalib.core.data_classes.FlatData` object under the new tracking
number, which `apply_flat_command` returns.

### 8. Analyse

```{code-block} python
from opticalib.ground.modal_decomposer import ZernikeFitter

fitter   = ZernikeFitter(fit_mask=flat.analysis_mask)
coeffs, resid = fitter.fit(az.frame(cube[0]), [2, 3, 4, 5])

ts = procedures.TimeSeries(interf, dm)
ts.acquire_time_series()
```

### 9. Plot

The bundled helpers know about masked pupils and DM geometry:

```{code-block} python
oplt.matshow(cube[0])              # masked phase map
oplt.surfshow(dm.get_shape())      # command vector as a mirror-shaped surface
oplt.cmdplot(flat.flat_cmd)        # command histogram over the actuator grid
```

## Working without hardware

Every device class in {doc}`/reference/devices` has a simulated twin in
{doc}`/reference/simulator` satisfying the same protocol, so the session above
runs offline:

```{code-block} python
from opticalib.simulator import AlpaoDm, Fake4DInterf

dm     = AlpaoDm(nacts=97)
interf = Fake4DInterf()

wf = interf.acquire_map()
wf.shape
```

This is the recommended way to develop analysis scripts, write regression
tests, and produce documentation figures.

## Running calpy non-interactively

The shell is IPython, so arguments after it can be forwarded:

```{code-block} bash
calpy -f ~/alpao_experiment -- -i myscript.py
```

Inside `myscript.py` you can rely on the preloaded namespace. The important
constraint is that **`AOCONF` must be set before `opticalib` is imported**,
because {mod}`opticalib.core.root` reads the configuration at import time and
builds the folder tree from it:

```{code-block} bash
# Works: calpy sets AOCONF, then imports opticalib
calpy -f ~/alpao_experiment -- -i myscript.py

# Also works, if you set it yourself first
export AOCONF=~/alpao_experiment/SysConfig/configuration.yaml
python myscript.py
```

To repoint a *running* process at a different experiment, call
{func}`~opticalib.core.root.set_configuration_file`, which re-reads the file and
updates the global `folders` object in place.

```{seealso}
- {doc}`/configuration` — every key in {{ yaml }}
- {doc}`/reference/procedures` — the stateful operations used above
- {doc}`/reference/dmutils` — capture planning, reduction and flattening
- {doc}`/reference/core` — folder tree, configuration API and data classes
```
