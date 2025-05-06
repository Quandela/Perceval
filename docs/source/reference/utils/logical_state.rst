Logical State
=============

A :code:`LogicalState` represents a pure qubit state. It is a list of 0s and 1s.

Their main purpose is to provide an easy way to convert a qubit state to a :code:`BasicState`.

They can be used in two ways:

- With :code:`Port`

    >>> import perceval as pcvl
    >>> encodings = [pcvl.Encoding.DUAL_RAIL, pcvl.Encoding.QUDIT2]
    >>> ports = [pcvl.Port(encoding, "my_name") for encoding in encodings]
    >>> ls = pcvl.LogicalState("101")
    >>> print(pcvl.get_basic_state_from_ports(ports, ls))
    |0,1,0,1,0,0>

- With :code:`Processor`, :code:`Experiment` or :code:`RemoteProcessor` that has Ports defined
  (recommended when using composition):

    >>> import perceval as pcvl
    >>> encodings = [pcvl.Encoding.DUAL_RAIL, pcvl.Encoding.QUDIT2]
    >>> e = pcvl.Experiment(6)
    >>> m = 0
    >>> for i, encoding in enumerate(encodings):
    >>>     e.add_port(m, pcvl.Port(encoding, f"{i}"))
    >>>     m += encoding.fock_length
    >>> ls = pcvl.LogicalState("101")
    >>> e.with_input(ls)
    >>> print(e.input_state)
    |0,1,0,1,0,0>

Note that the way a :code:`LogicalState` is converted depends on the encoding,
and the number of modes, photons, and the expected number of qubits is only guaranteed by the encoding,
not by the conversion itself.

.. note::

  The perceval convention for LogicalStates is that the first digit is represented in the first mode(s) of a circuit.

.. autoclass:: perceval.utils.logical_state.LogicalState
  :members:

.. autofunction:: perceval.utils.logical_state.generate_all_logical_states
