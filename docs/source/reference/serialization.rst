serialization
^^^^^^^^^^^^^

Perceval provides a generic way to serialize most objects into strings that can be deserialized later on
to get back the original object.

>>> import perceval as pcvl
>>> from perceval.serialization import serialize, deserialize  # Note: this is not directly at perceval's root
>>> s = serialize(pcvl.Circuit(3).add(0, pcvl.BS()).add(1, pcvl.BS()))
>>> print(s)
:PCVL:zip:eJyzCnAO87FydM4sSi7NLLFydfTN9K9wdI7MSg52DsyO9AkNCtWu9DANqMj3cg50hAPP9GwvBM+xEKgWwXPxRFWbZeDpGERdMwHhijWy
>>> c = deserialize(s)  # Creates a copy of the circuit from the string representation

Serialize
=========

Most perceval objects can be serialized. This includes (but is not limited to):

- Circuit and basic components (PS, BS, ...)
- Experiments
- BasicState and StateVector
- Matrix
- Heralds
- Ports
- BSDistribution, SVDistribution, BSCount, BSSamples
- NoiseModel
- PostSelect
- Detector

.. note::
  Some python containers (list, dict) are serialized recursively, so the returned value is a container of strings
  (or a container of containers of ... of strings).
  This allows a more simple serialization and deserialization later on using the JSON format,
  but implies that the returned type will depend on the input type.

Note however that some objects can't be serialized. This includes (but is not limited to):

- Algorithm
- Processor and RemoteProcessor (the experiment within them can however be serialized)

.. warning::
  Non-serializable objects will be silently returned as they are by the :meth:`serialize()` method,
  which can produce errors for example if trying to save the result later on.

The :meth:`serialize()` method has an optional kwarg argument :code:`compress` that can be either a boolean or a list of types
on which to apply the compression (which can be useful for containers).
The default value of this parameter depends on the object to serialize.

If :code:`compress` is True, or for types that match, a compression will be applied to try to reduce the string size.
Note that if :code:`compress` is False, some string representations will be human readable (such as BasicState),
and the object type will always be specified by a prefix qualifier.

Also, the :code:`serialization` module has a :meth:`serialize_to_file()` method that takes a file path and an object.
Note that this method will fail if any of the objects is not serializable either by perceval or by JSON.

Deserialize
===========

The deserialization part of the process is more straightforward than the serialization part
as any class that can be serialized can be deserialized.

As such, using :meth:`deserialize()` on an object returned by :meth:`serialize()` should always produce a copy of the
initial object, or the object itself if it is not serializable.

.. note::
  Using serialization to make copies inside a single code instance is generally a bad idea as it is expected to be slow
  compared to a direct copy of the objects.

There is also a :meth:`deserialize_file()` method that can deserialize anything that was previously stored using
:meth:`serialize_to_file()` using the same file path.
