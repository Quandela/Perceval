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

Perceval is conceived as an object-oriented modular Python framework organized around the following elements:

* Tools to :ref:`build linear optical circuits <Circuits>` from a collection of pre-defined :ref:`components <Unitary Components>`
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

    @article{heurtel2023perceval,
    doi = {10.22331/q-2023-02-21-931},
    url = {https://doi.org/10.22331/q-2023-02-21-931},
    title = {Perceval: {A} {S}oftware {P}latform for {D}iscrete {V}ariable {P}hotonic {Q}uantum {C}omputing},
    author = {Heurtel, Nicolas and Fyrillas, Andreas and Gliniasty, Gr{\'{e}}goire de and Le Bihan, Rapha{\"{e}}l and Malherbe, S{\'{e}}bastien and Pailhas, Marceau and Bertasi, Eric and Bourdoncle, Boris and Emeriau, Pierre-Emmanuel and Mezher, Rawad and Music, Luka and Belabas, Nadia and Valiron, Benoît and Senellart, Pascale and Mansfield, Shane and Senellart, Jean},
    journal = {{Quantum}},
    issn = {2521-327X},
    publisher = {{Verein zur F{\"{o}}rderung des Open Access Publizierens in den Quantenwissenschaften}},
    volume = {7},
    pages = {931},
    month = feb,
    year = {2023}
   }

Related Projects
================

Perceval is used in several higher-level projects:

- **perceval-interop**: Interoperability tools for conversion between photonic and gate based Quantum computing. See the project here: `perceval-interop <https://perceval.quandela.net/interopdocs/>`_.
- **Merlin**: A tool to bring quantum computing to AI practitioners, requiring no prior quantum expertise.. Learn more here: `merlin <https://merlinquantum.ai/>`_.


.. toctree::
   :caption: Documentation
   :maxdepth: 2
   :hidden:

   basics
   usage
   circuits
   polarization
   backends
   tools
   legacy

.. toctree::
   :caption: Beginner tutorials
   :maxdepth: 2
   :hidden:

   notebooks/Walkthrough-cnot
   notebooks/Tutorial

.. toctree::
   :caption: Advanced tutorials
   :maxdepth: 2
   :hidden:

   notebooks/BS-based_implementation
   notebooks/LOv_rewriting_rules
   notebooks/Simulation_non-unitary_components
   notebooks/Remote_computing
   notebooks/Graph_States_generation_and_display
   notebooks/Tomography_walkthrough
   notebooks/Density_matrix_Fock_space
   notebooks/Quantum_teleportation_feed_forward

.. toctree::
   :caption: Boson sampling
   :maxdepth: 2
   :hidden:

   notebooks/Boson_sampling
   notebooks/MPS_techniques_for_boson_sampling

.. toctree::
   :caption: Standard quantum algorithms
   :maxdepth: 2
   :hidden:

   notebooks/Shor_Implementation
   notebooks/2-mode_Grover_algorithm

.. toctree::
   :caption: Variational quantum algorithms
   :maxdepth: 2
   :hidden:

   notebooks/Differential_equation_resolution
   notebooks/Variational_Quantum_Eigensolver
   notebooks/Reinforcement_learning
   notebooks/QUBO
   notebooks/QLOQ_QUBO_tutorial

.. toctree::
   :caption: Quantum walk
   :maxdepth: 2
   :hidden:

   notebooks/Two-particle_bosonic-fermionic_quantum_walk

.. toctree::
   :caption: Others
   :maxdepth: 2
   :hidden:

   notebooks/Gedik_qudit
   notebooks/Boson_Bunching
   notebooks/quantum_kernel_methods

.. toctree::
   :caption: Code Reference
   :maxdepth: 2
   :hidden:

   reference/algorithm/index
   reference/backends/index
   reference/components/index
   reference/error_mitigation
   reference/providers
   reference/rendering/index
   reference/runtime/index
   reference/serialization
   reference/simulators/index
   reference/utils/index
   reference/utils_algorithms/index
   reference/logging
   reference/exqalibur/index
   reference/statevector
   reference/qiskit_converter

.. toctree::
   :caption: Community

   contributing
   bibliography
