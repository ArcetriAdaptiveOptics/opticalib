# OptiCalib : adaptive OPTics package for deformable dirrors CALibration

![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/pietroferraiuolo/labott/tests.yml)

`Adaptive Optics Group, INAF - Osservatorio Astrofisico di Arcetri`  

- [Pietro Ferraiuolo](mailto:pietro.ferraiuolo@inaf.it)

- [Marco Xompero](mailto:marco.xompero@inaf.it)

OptiCalib is a python package which first goal is to make easy deformable mirror's calibration in the laboratory (and not only).

It was born as a general extrapolation of the software built for the control and calibration of the `ELT @ M4` adaptive mirror and it's calibration tower, `OTT` (`Optical Test Tower`).

## Description

The `OPTICALIB` package serves two main purposes:

- Making connection to the hardware (interferometers, DMs, ...) easy;

- Providing routines for Deformable Mirrors calibrations.

It can be installed from the repository, while waiting for an official release, as

```bash
pip install git+"https://github.com/pietroferraiuolo/labott.git"
```

The software will create an entry point script called `calpy`, which is usefull to set up a specific experiment's environment. Let's say we have an optical bench composed of an interferometer 4D PhaseCam6110 and an Alpao Deformable mirror, say DM820. We can create the experiment's environment just like:

```bash
calpy -f ~/alpao_experiment --create
```

This will create, in the `~/alpao_experiment` folder, the package's data folder tree, together with a configuration file in the `SysConfig` folder. The [configuration file](./opticalib/core/_configurations/configuration.yaml), documented [here](./opticalib/core/_configurations/DOCS.md), is where all devices must be specified.

Once done with the configuration, we can then start using out instruments:

```bash
calpy -f ~/alpao_experiment
```

```python
# The `calpy` function will automatically import opticalib (with `opt` as alias), as well as the `opticalib.dmutils` as dmutils

interf = opt.PhaseCam(6110) # set in the configuration file
dm = opt.AlpaoDm(820)       # set in the configuration file

# Having the bench set up and the configuration file set, we can acquire an Influence Function by just doing

tn = dmutils.iff_module.iffDataAcquisition(dm, interf) # Optional paramenters are `modesList, modesAmplitude, template`, which if not specified are read from the configuration file
```

## Documentation
For the API references, check [here](docs/opticalib.pdf) (work in progress...), while for the configuration file documentation check [here](./opticalib/core/_configurations/DOCS.md)
