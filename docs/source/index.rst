:github_url: https://github.com/Quandela/Perceval

.. figure:: _static/img/perceval.jpg
   :align: right
   :width: 300

   Perceval, *The Story of the Grail* (Chrétien de Troyes, circa 1180)

Welcome to the Perceval's documentation!
========================================

Through a simple object-oriented Python API, Perceval provides tools for composing circuits from linear optical components,
defining single-photon sources, manipulating Fock states, running simulations, reproducing published experimental papers and
experimenting with a new generation of quantum algorithms. It aims to be a companion tool for developing photonic circuits –
for simulating and optimising their design, modelling both the ideal and realistic behaviours, and proposing a normalised
interface to control them through the concept of *backends*.

Perceval is conceived as an object-oriented modular Python framework orgainised around the following elements:

* Tools to :ref:`build linear circuits <Circuits>` with a collection of pre-defined :ref:`components <Components>`
* Powerful :ref:`computing backends <Computing Backends>` implemented in C++
* Misc technical utilities to manipulate:
   - :ref:`Fock states, state vectors and state vector distribution <States>`,
   - :ref:`unitary matrices <Matrices>`,
   - :ref:`parameters <Parameters>`
Transversal tools for flexible :ref:`visualization <Visualization>` of the circuits and results which are compatible
with notebooks or local development environments

Perceval has been developed as a complete toolkit for physicists and computer scientists, and for students, researchers,
and practitioners of quantum computing.

If you are using Perceval for academic work, please cite the `following paper in publication <https://perceval.quandela.net/Perceval-Whitepaper.pdf>`_ with the following reference:

.. code:: latex

    @article{perceval_white_paper,
      title = {Perceval: {A} {Software} {Platform} for {Discrete} {Variable} {Photonic} {Quantum} {Computing}},
      url = {https://perceval.quandela.net/Perceval-Whitepaper.pdf},
      author = {Heurtel, Nicolas and Fyrillas, Andreas and {{d}e {G}liniasty}, Grégoire and {{L}e {B}ihan}, Raphaël and Malherbe, Sébastien and Pailhas,
      Marceau and Bourdoncle, Boris and Emeriau, Pierre-Emmanuel and Mezher,
      Rawad and Music, Luka and Belabas, Nadia and Valiron, Benoît and Senellart,
      Pascale and Mansfield, Shane and Senellart, Jean},
      year = {2022},
    }


.. toctree::
   :caption: Documentation
   :maxdepth: 2
   :hidden:

   basics
   usage
   circuits
   states
   polarization
   backends
   components
..   tools

.. toctree::
   :caption: Examples
   :maxdepth: 2
   :hidden:

   notebooks/walkthrough-cnot
   notebooks/Differential equation solving
   notebooks/Shor Implementation
   notebooks/Boson Sampling
   notebooks/Variational Quantum Eigensolver
   notebooks/2-mode Grover algorithm.ipynb

.. toctree::
   :caption: Code Reference
   :maxdepth: 2
   :hidden:

   reference/statevector
   reference/polarization
   reference/circuit
   reference/utils

.. toctree::
   :caption: Community

   contributing
   bibliography
