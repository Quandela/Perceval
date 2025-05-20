Experiment
^^^^^^^^^^

Experiments behave the same way as Processors and RemoteProcessors,
except that they don't know how or where to simulate them (they don't have backend or platform);
they simply describe the elements of an optical table and the post-processing rules.

>>> import perceval as pcvl
>>> e = pcvl.Experiment(2, noise=pcvl.NoiseModel(0.8), name="my experiment").add(0, pcvl.BS())
>>> e.add_herald(0, 1)
>>> p = pcvl.Processor("SLOS", e)
>>> rp = pcvl.RemoteProcessor("sim:slos").add(e)

Experiments have two main purposes that :code:`Processor` and :code:`RemoteProcessor` can't fulfill:

- They can be used to create several Processors describing the same experiment with different backends.
- They can be serialized using perceval serialization, so they can be stored and retrieved easily.

>>> from perceval.serialization import serialize, deserialize
>>> e_str = serialize(e)  # This is a regular string
>>> e_copy = deserialize(e_str)

.. autoclass:: perceval.components.experiment.Experiment
   :members:
