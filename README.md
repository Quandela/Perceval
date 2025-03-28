[![GitHub release](https://img.shields.io/github/v/release/Quandela/Perceval.svg?style=plastic)](https://github.com/Quandela/Perceval/releases/latest)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/perceval-quandela?style=plastic)
[![CI](https://github.com/Quandela/Perceval/actions/workflows/python-publish.yml/badge.svg)](https://github.com/Quandela/Perceval/actions/workflows/python-publish.yml)

[![CI](https://github.com/Quandela/Perceval/actions/workflows/autotests.yml/badge.svg)](https://github.com/Quandela/Perceval/actions/workflows/autotests.yml)
[![CI](https://github.com/Quandela/Perceval/actions/workflows/build-and-deploy-docs.yml/badge.svg)](https://github.com/Quandela/Perceval/actions/workflows/build-and-deploy-docs.ym)

# Perceval <a href="https://perceval.quandela.net" target="_blank"> <img src="logo-perceval.png" width="50" height="50"> </a>



Through a simple object-oriented python API, Perceval provides tools for building a circuit with linear optics components,
defining single-photon sources and their error model, manipulating Fock states, running simulations, reproducing published experimental papers results,
and experimenting with a new generation of quantum algorithms.

It is interfaced with the available QPUs on
the [Quandela cloud](https://cloud.quandela.com/webide/), so it is possible to run computations on an
actual photonic computer.

Perceval aims to be a companion tool for developing discrete-variable photonics circuits
- while simulating their design, modelling their ideal and real-life behaviour;
- and proposing a normalized interface to control photonic quantum computers;
- while using powerful simulation backends to get state-of-the-art simulation;
- and also allowing direct access to the QPUs of Quandela.

Perceval has been developed as a complete toolkit for physicists and quantum computational students, researchers and
practitioners.

# Key Features

* Powerful Circuit designer making use of predefined components
* Simple python API and powerful simulation backends optimized in C
* Misc technical utilities to manipulate State Vector, Unitary Matrices, and circuit Parameters
* Transversal tools for visualization compatible with notebooks or local development environments
* Numerical and symbolical computation
* Modular architecture welcoming contributions from the community

# Installation

Perceval requires:

* Python between 3.9 and 3.13

## PIP
We recommend installing it with `pip`:

```bash
pip install --upgrade pip
pip install perceval-quandela
```

If you are looking to use our qiskit, qutip, cqasm or myqlm bridges you can specify it:
```bash
pip install perceval-quandela[qiskit_bridge] #install qiskit and seaborn
pip install perceval-quandela[qutip_bridge] #install qutip
pip install perceval-quandela[myqlm_bridge] #install myqlm
pip install perceval-quandela[cqasm_bridge] #install cqasm
pip install perceval-quandela[all] #install all above
```

## GitHub
```bash
git clone https://github.com/quandela/Perceval
```
then to install Perceval:
```bash
pip install .
```
Or for developers:
```bash
pip install -e .
```
Or to use converters:
```bash
pip install .[all]
```

# Running tests and benchmarks

Unit tests files are part of the repository in `tests/` and can be run with:

```
pip install -r tests/requirements.txt
pytest
```

Additionally, you can see a coverage report with the command:

```
pytest --cov=perceval
```

Benchmark tests for computing-intensive functions are in `benchmark/` and can be run with:

```
pytest benchmark/benchmark_*.py
```

Comparison benchmarks for different platforms are also committed in `.benchmarks/` - see [pytest-benchmark documentation](https://pytest-benchmark.readthedocs.io/en/stable/usage.html) for more information.

# Documentation and Forum

* The [documentation](https://perceval.quandela.net/docs)
* The [Community Forum](https://perceval.quandela.net/forum)

#
[<img src="logo-quandela.png" width="300" height=auto>](https://www.quandela.com/)

[![Twitter Follow](https://img.shields.io/twitter/follow/Quandela_SAS?style=social)](https://twitter.com/Quandela_SAS)
[![YouTube Channel Subscribers](https://img.shields.io/youtube/channel/subscribers/UCl5YMpSqknJ1n-IT-XWfLsQ?style=social)](https://www.youtube.com/channel/UCl5YMpSqknJ1n-IT-XWfLsQ)
