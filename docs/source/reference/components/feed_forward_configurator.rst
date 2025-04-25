Feed forward Configurators
==========================

Configurators are the way to perform a feed-forward computation.
As they are non-unitary components, they can only be added to :code:`Processor` instances.

Their purpose is to link measurements on given modes to circuits to configure.

They are based on the following common features:

* A default circuit is mandatory for when the measurement does not fall into one of the configured cases. This circuit determines the size of the configured circuit.
* All referenced circuits have the same size.
* The measured modes must be classical (placed after a detector).
* The circuit position is determined by an offset between the configurator and the configured circuit.

This offset represents the number of modes between the measured modes and the circuit. For positive offsets, the circuit is placed below, the offset being the number of empty modes between the configurator and the circuit (0 means the circuit is right below the configurator). For negative values, the circuit is placed above the measured modes (-1 means that the circuit is right above the configurator).

Two configurators exist.

FFCircuitProvider class reference
---------------------------------

This class directly links measurements to circuits or processors.
Any circuit or processor matching the default circuit size can be used given all parameters have numerical values.

>>> import perceval as pcvl
>>> p = pcvl.Processor("SLOS", 4)
>>> c = pcvl.FFCircuitProvider(1, offset=1, default_circuit=pcvl.Circuit(2), name="FFCircuitProvider Example")
>>> c.add_configuration([1], pcvl.BS())
>>> p.add(0, pcvl.Detector.threshold())
>>> p.add(0, c)

.. autoclass:: perceval.components.feed_forward_configurator.FFCircuitProvider
   :members:
   :inherited-members:

FFConfigurator class reference
------------------------------

This class links measurements to a mapping of parameter values that can be set in the given circuit.

>>> import perceval as pcvl
>>> p = pcvl.Processor("SLOS", 4)
>>> phi = pcvl.P("phi")
>>> c = pcvl.FFConfigurator(2, offset=1, controlled_circuit=pcvl.PS(phi), default_config={"phi": 0}, name="FFConfigurator Example")
>>> c.add_configuration([1, 0], {"phi": 1.23})
>>> p.add(0, pcvl.Detector.threshold())
>>> p.add(1, pcvl.Detector.threshold())
>>> p.add(0, c)

.. autoclass:: perceval.components.feed_forward_configurator.FFConfigurator
   :members:
   :inherited-members:
