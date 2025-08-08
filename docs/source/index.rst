:github_url: https://github.com/Quandela/Perceval

.. figure:: _static/img/perceval.jpg
   :align: right
   :width: 250
   :figwidth: 250
   :alt: Extract from Chrétien de Troyes

   Perceval, the Story of the Grail – Chrétien de Troyes (circa 1180)

Welcome to the Perceval documentation!
======================================

Perceval is an open source linear optics quantum framework. It provides a powerful language to describe linear optics
setups through a simple object-oriented API, and is able to simulate them and send computation requests to remote
Quantum Processing Units (QPU) and simulators.

* To start using Perceval, see: :ref:`Getting started`
* To contribute to Perceval, see: :ref:`Welcoming Contributors`

Perceval has been developed as a complete toolkit for physicists, computer scientists, students, researchers,
and practitioners of quantum computing. It can be used to reproduce published experimental works or to experiment
directly with a new generation of quantum algorithms.

If you are using Perceval for academic work, please cite the `Perceval white paper <https://arxiv.org/abs/2204.00602>`_
as:

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

Perceval is used in several higher-level projects (non-exhaustive list):

* **perceval-interop**: Interoperability tools for conversion between photonic and gate based Quantum computing. See
  the project here: `perceval-interop <https://perceval.quandela.net/interopdocs/>`_.
* **MerLin**: A framework to bring quantum computing to AI practitioners, requiring no prior quantum expertise.
  Learn more here: `merlinquantum <https://merlinquantum.ai/>`_.


.. toctree::
   :maxdepth: 2
   :hidden:

   getting_started
   tutorial_beginner
   tutorial_advanced
   tutorial_expert
   legacy

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
   reference/exqalibur/index

.. toctree::
   :caption: Examples
   :maxdepth: 2
   :hidden:

   examples_boson_sampling
   examples_standard_algo
   examples_vqa
   examples_quantum_walk
   examples_others

.. toctree::
   :caption: Community

   contributing
   bibliography
