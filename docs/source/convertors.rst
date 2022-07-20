Convertors
==========

The ``convertor`` package contain classes aimed at converting circuits and data from and to perceval standards.

Qiskit Convertor
----------------

Overview
^^^^^^^^

.. note::
  Qiskit is not listed as a Perceval dependency. Users wishing to use Qiskit shall install it by themselves.

`Qiskit <https://qiskit.org/>`_ is an opensource quantum development library. A Qiskit ``QuantumCircuit`` can be
converted to an equivalent photonic circuit using Perceval ``QiskitConverter``.

>>> import qiskit
>>> from perceval.converters import QiskitConverter
>>> import perceval.lib.phys as phys

Create a Quantum Circuit (the following is pure Qiskit syntax):

>>> qc = qiskit.QuantumCircuit(2)
>>> qc.h(0)
>>> qc.cx(0, 1)
>>> print(qc.draw())
     ┌───┐
q_0: ┤ H ├──■──
     └───┘┌─┴─┐
q_1: ─────┤ X ├
          └───┘

Then convert the Quantum Circuit with Perceval QiskitConvertor:

>>> qiskit_convertor = QiskitConverter(phys)
>>> perceval_processor = qiskit_convertor.convert(qc)

See also:
`Qiskit tutorial <https://qiskit.org/documentation/tutorials/circuits/1_getting_started_with_qiskit.html>`_

Parameters
^^^^^^^^^^

``QiskitConverter`` constructor usage:

>>> converter = QiskitConverter(library, source=None)

With:

* ``library``: any Perceval component library such as ``phys`` or ``symb``
* ``source``: a Perceval Source object representing a photonic source (default is a perfect source)

``convert`` method usage:

>>> processor = qiskit_convertor.convert(qc: qiskit.QuantumCircuit, heralded: bool = None)

With:

* ``qc``: the qiskit.QuantumCircuit to be converted
* ``heralded``: bool (default = None)

  * True => use only heralded CNOT
  * False => use only post-processed CNOT
  * None (not set) => use heralded CNOT for all gates except the last one (which is post-processed)

* **Returns**: a ``Processor`` containing the ``qiskit.QuantumCircuit`` equivalent, heralds and post
selection function.
