Catalog
=======

The concept of the *catalog* is to provide the user a set of basic of qbit or photonic gate or circuit.

All circuits share the same interface that is describe by the following class ``CatalogItem``:

.. autoclass:: perceval.components.component_catalog.CatalogItem
   :members:

The catalog object work as a dictionary. To select the wanted component you must address it with its catalog key.

For example to get an heralded CZ gate, you must call it as followed:

.. code-block:: Python

   from perceval import catalog
   catalog['heralded cz']

You can after either get it as a circuit or a processor:

.. code-block:: Python

   from perceval import catalog
   processor = catalog['heralded cz'].build_experiment() # Will return an experiment
   circuit = catalog['heralded cz'].build_circuit() # Will return a circuit

If a gate have parameters, like for instance a Mach-Zehnder interferometer phase first you can set those parameters as followed:

.. code-block:: Python

   import math
   from perceval import catalog
   circuit = catalog["mzi phase first"].build_circuit(phi_a=math.pi,
                                                      phi_b=math.pi))

.. include:: ../../../build/catalog.rst
