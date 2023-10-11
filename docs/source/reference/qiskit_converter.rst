Qiskit converter
================

Overview
--------

.. note::
  Qiskit is not listed as a Perceval dependency. Users wishing to use Qiskit shall install it by themselves.

`Qiskit <https://qiskit.org/>`_ is an opensource quantum development library. A Qiskit ``QuantumCircuit`` can be
converted to an equivalent Perceval ``Processor`` using ``QiskitConverter``.

Minimal code:

>>> import qiskit
>>> from perceval.converters import QiskitConverter
>>> from perceval.components import catalog
>>> # Create a Quantum Circuit (the following is pure Qiskit syntax):
>>> qc = qiskit.QuantumCircuit(2)
>>> qc.h(0)
>>> qc.cx(0, 1)
>>> print(qc.draw())
     ┌───┐
q_0: ┤ H ├──■──
     └───┘┌─┴─┐
q_1: ─────┤ X ├
          └───┘
>>> # Then convert the Quantum Circuit with Perceval QiskitConvertor
>>> qiskit_convertor = QiskitConverter(catalog)
>>> perceval_processor = qiskit_convertor.convert(qc)

See also:

`Qiskit tutorial <https://qiskit.org/documentation/tutorials/circuits/1_getting_started_with_qiskit.html>`_

:ref:`Qiskit conversion and decomposition example<Conversion from Qiskit Circuit>`

Class reference
---------------

.. autoclass:: perceval.converters.qiskit_converter.QiskitConverter
   :members:
   :inherited-members:
