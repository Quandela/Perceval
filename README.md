# Perceval

Through a simple object-oriented python API, Perceval provides tools for building a circuit with linear optics components,
defining single-photon source, manipulating Fock states, running simulation, reproducing published experimental papers
and experiment new generation of quantum algorithms. It aims to be a companion tool for developing photonics circuits
- while simulating on their design, modeling their ideal and real-life behaviour, and proposing a normalized interface
- to control them through the concept of backends.

Perceval has been developed as a complete toolkit for physicists and quantum computational students, researchers and
practitioners.

# Key Features

* Powerful Circuit designer making use of predefined components
* Simple python API and powerful simulation backends optimized in C
* Misc technical utilities to manipulate State Vector, Unitary Matrices, and circuit Parameters
* Transversal tools for visualization compatible with notebooks or local development environments
* Works numerically and symbolically
* Modular architecture welcoming contributions from the community

# Installation

Perceval requires:

* Python 3.6 or above

We recommend installing it with `pip`:

```bash
pip install --upgrade pip
pip install perceval-quandela
```

or simply from github:

```bash
git clone https://github.com/quandela/Perceval
python setup.py install # [or 'develop' for developpers]
```

# Documentation and Forum

* The [documentation](https://perceval.quandela.net/docs)
* The [Community Forum](https://perceval.quandela.net/forum)
