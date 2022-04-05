:github_url: https://github.com/Quandela/Perceval

.. figure:: _static/img/perceval.jpg
   :align: right
   :width: 250
   :figwidth: 250
   :alt: Extract from Chrétien de Troyes

   Perceval, the Story of the Grail – Chrétien de Troyes (circa 1180)

Welcome to the Perceval documentation!
======================================

Through a simple object-oriented Python API, Perceval provides tools for composing photonic circuits from linear optical components like beamsplitters and phase shifters,
defining single-photon sources, manipulating Fock states, and running simulations.

Perceval can be used to reproduce published experimental works or to
experiment directly with a new generation of quantum algorithms.

It aims to be a companion tool for developing photonic circuits –
for simulating and optimising their design,
modelling both the ideal and realistic behaviours, and proposing a normalised
interface to control them through the concept of *backends*.

Perceval is conceived as an object-oriented modular Python framework orgainised around the following elements:

* Tools to :ref:`build linear optical circuits <Circuits>` from a collection of pre-defined :ref:`components <Components>`
* Powerful :ref:`computing backends <Computing Backends>` implemented in C++
* A variety of technical utilities to manipulate:
   - :ref:`Fock states, state vectors and state vector distributions <States>`,
   - :ref:`unitary matrices <Matrices>`,
   - :ref:`parameters <Parameters>`.
It also includes transversal tools for flexible :ref:`visualization <Visualization>` of the circuits and results which are compatible
with notebooks or local development environments.

Perceval has been developed as a complete toolkit for physicists and computer scientists, and for students, researchers,
and practitioners of quantum computing.

If you are using Perceval for academic work, please cite the `Perceval white paper <https://arxiv.org/abs/2204.00602>`_ as:

.. code:: latex

    @article{heurtel_perceval_2022,
    title = {Perceval: {A} {Software} {Platform} for {Discrete} {Variable} {Photonic} {Quantum} {Computing}},
    author = {Heurtel, Nicolas and Fyrillas, Andreas and de Gliniasty, Grégoire and Bihan, Raphaël Le and Malherbe, Sébastien and Pailhas, Marceau and Bourdoncle, Boris and Emeriau, Pierre-Emmanuel and Mezher, Rawad and Music, Luka and Belabas, Nadia and Valiron, Benoît and Senellart, Pascale and Mansfield, Shane and Senellart, Jean},
    doi = {10.48550/ARXIV.2204.00602},
    url = {https://arxiv.org/abs/2204.00602},
    keywords = {Quantum Physics (quant-ph), Computational Physics (physics.comp-ph), FOS: Physical sciences, FOS: Physical sciences},
    publisher = {arXiv},
    year = {2022}
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
