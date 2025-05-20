Ports and data encoding
^^^^^^^^^^^^^^^^^^^^^^^

Ports are a mean to describe data encoding at the input and the output of a circuit in an :ref:`Experiment`.

Encoding
========

.. autoenum:: perceval.utils._enums.Encoding
   :members:

PortLocation
=============

.. autoenum:: perceval.components.port.PortLocation
   :members:

Port
====

.. autoclass:: perceval.components.port.Port
   :members:

Herald
======

.. autoclass:: perceval.components.port.Herald
   :members:

Utilitary functions
===================

.. autofunction:: perceval.components.port.get_basic_state_from_ports

.. autofunction:: perceval.components.port.get_basic_state_from_encoding
