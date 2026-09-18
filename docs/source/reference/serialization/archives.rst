Serialization
=============

Most of the Serialization process uses Archives, which describe every recorded object before transforming them into a storable format.

The archive API serializes objects whose exact classes are known to the serialization registry.
The standard Python container types and supported Perceval classes are registered when Perceval is imported.
It is possible to add your own classes to the registry (see :ref:`Registering new classes`).

Serializing an object
^^^^^^^^^^^^^^^^^^^^^

Create an :class:`~perceval.serialization.OutputArchive`, then add an object with
:meth:`~perceval.serialization.Serialization.serialize`:

>>> import perceval as pcvl
>>> from perceval.serialization import OutputArchive, Serialization
>>>
>>> circuit = pcvl.Circuit(2) // pcvl.BS.H()
>>> output_archive = OutputArchive()
>>> Serialization.serialize(circuit, output_archive)
>>> serialized_circuit = output_archive.to_text()

The returned string contains the archive format version, its root objects, and the registered type tags needed to
reconstruct the objects. Treat this representation as an opaque string: use the archive API instead of parsing or
editing it directly.

Set ``compress=True`` to produce a compressed text representation:

>>> compressed_circuit = output_archive.to_text(compress=True)

Both representations can be passed to :meth:`~perceval.serialization.InputArchive.from_text`.

Deserializing an object
^^^^^^^^^^^^^^^^^^^^^^^

Build an :class:`~perceval.serialization.InputArchive` from the stored text, then retrieve the object with
:meth:`~perceval.serialization.Serialization.deserialize`:

>>> from perceval.serialization import InputArchive
>>>
>>> input_archive = InputArchive.from_text(serialized_circuit)
>>> restored_circuit = Serialization.deserialize(input_archive)
>>> type(restored_circuit) is pcvl.Circuit
True
>>> restored_circuit.m
2

.. note::
   The legacy :meth:`perceval.serialization.deserialize` method is able to read from an archive string.
   In that case, it returns the first serialized object.

Serializing several objects
^^^^^^^^^^^^^^^^^^^^^^^^^^^

An archive can contain more than one root object. Call ``Serialization.serialize`` once for each object, then call
``Serialization.deserialize`` in the same order:

>>> state = pcvl.BasicState("|1,0>")
>>> output_archive = OutputArchive()
>>> Serialization.serialize(circuit, output_archive)
>>> Serialization.serialize(state, output_archive)
>>>
>>> input_archive = InputArchive.from_text(output_archive.to_text())
>>> restored_circuit = Serialization.deserialize(input_archive)
>>> restored_state = Serialization.deserialize(input_archive)

Deserializing consumes one root from the input archive.
``len(input_archive)`` returns the number of roots that have not yet been consumed.

Object identity
^^^^^^^^^^^^^^^

Objects are memoized inside an archive. When the same mutable object appears several times in the serialized object
graph, every occurrence points to the same restored instance:

>>> shared = [1, 2]
>>> value = [shared, shared]
>>> output_archive = OutputArchive()
>>> Serialization.serialize(value, output_archive)
>>> restored = Serialization.deserialize(InputArchive.from_text(output_archive.to_text()))
>>> restored[0] is restored[1]
True

This memoization also allows registered serializers to handle cyclic object graphs.

API reference
^^^^^^^^^^^^^

.. autoclass:: perceval.serialization.OutputArchive
  :members:
  :inherited-members:

.. autoclass:: perceval.serialization.InputArchive
  :members:
  :inherited-members:

.. automethod:: perceval.serialization.Serialization.serialize

.. automethod:: perceval.serialization.Serialization.deserialize
