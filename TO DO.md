# Changes to be made in the `XuPy Transition` branch

```md
✅ GROUND
 |__ ✅ `computerec.py` : rewrite the module, performing only the operations when asked and with caching and gpu support
 |__ ✅ `zernike.py` : Enable caching and gpu support of the fitting functions
 |__ ✅ `osutils.py` : Enable array on GPU support for I/O operations (save/read fits)
 |                      Example: read fits as xupyMaskedArrays (flag)
 |
✅ DMUTILS
 |__ ✅ `iff_processing.py` : Enable GPU support for image processing
 |__ ✅ `flattening.py` : make CmdMat, IntMat e RecMat pubblicly available
 |
[] `analyzers.py` : Check for GPU useful operations
 |__ ✅ modify and/or rewrite some functions
 |__ [] Clarify what to do with these functions:
 |       - `frame()` : useless
 |       - `spectrum()` : seems done for SPLATT...
 |       - `frame2Ottframe` : clearly for M4 -> transfer it
 |       - `timevec` : what is it usefull for? (from a GENERAL PoV)
 |       - `track2jd` : same
 |       - `track2date` : same
 |       - `readTemperature` : was for M4, why here?
 |       - `readZernike` : same
 |       - `zernikePlot` : it's a fit on a cube...
 |
✅ `alignment.py` : Check for GPU useful operations
 |__ ✅ `_push_pull_redux` : copy that of `iff_processing` -> actually abstracted and put into `analyzers.py`
 |__ ✅ `zern_routine` + `global_zern_on_roi` : rewrite into the more general way, using the `ZernikeFitter`

```

## Performance gains considerations

### `iff_processing` module

$\bar{t_e}$ = Average Execution Time; **i** = initial ; **f** = final (after optimization)

| function | $\bar{t_e^i}$ [s] | $\bar{t_e^f}$ [s] | gain |changes | notes | GPU |
| -------- | --------------- | -------------- | ------- | ----- | --- | --- |
|`iffRedux` | 41.2 | 14.9 | ~63%  | Computation vectorization, GPU, I/O parallelization and prefetching | Test done with DP data *tn=20250911_110614*: ~1200 images to be processed. I/O sweet spot performance reached with n_workers = 8, 1 modes prefetch | NVIDIA RTX 5080 Mobile |
| `filterZernikeCube` | 43 | 22 | ~50%  | Fitting algorithm optimization with little GPU support | modes fitted: [1,2,3] | NVIDIA RTX 5080 Mobile |
