conversion
^^^^^^^^^^

Perceval provides helper methods to convert the three types of results of the :ref:`Sampler`
(namely the :ref:`BSDistribution`, :ref:`BSCount` and :ref:`BSSamples`) into each other.

>>> import perceval as pcvl
>>> distribution = pcvl.BSDistribution({pcvl.BasicState([1, 0]): 0.4, pcvl.BasicState([0, 1]): 0.6})
>>> print(pcvl.probs_to_sample_count(distribution, count=1000))  # Sampling noise is applied; results may vary
{
  |1,0>: 392
  |0,1>: 608
}

Note that for methods converting from probabilities, passing a kwarg with a count is mandatory.
This can either be:

- :code:`count`, in which case this is the number of resulting samples.
- :code:`max_shots` and/or :code:`max_samples`, in which case the one defined or the minimum of the two is the number of resulting samples.

Conversion code reference
=========================

.. automodule:: perceval.utils.conversion
   :members:
