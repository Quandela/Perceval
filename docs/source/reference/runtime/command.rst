Command
^^^^^^^

A :code:`Command` describes a type of method through a name and an ordered parameter signature. It does
not perform any work itself: a :ref:`Computation` associates it with an experiment, and a :ref:`Computer` implements
the command.

The signature is a list of :code:`(name, expected_type, mandatory)` tuples. For example, a command requiring an integer
:code:`count` and accepting an optional string :code:`label` can be defined as follows:

>>> import perceval as pcvl
>>>
>>> command = pcvl.Command(
...     "custom_command",
...     [("count", int, True), ("label", str, False)],
... )

Set the expected type to :code:`None` to accept a value of any type. The order of the signature determines how
positional arguments are assigned.

Filling and checking parameters
===============================

The :meth:`fill()` method maps positional and keyword arguments to the signature and checks argument names and types:

>>> command.fill(10, label="example")
{'count': 10, 'label': 'example'}

It raises :class:`TypeError` for too many positional arguments, unknown or duplicate names, and values of the wrong
type. Any parameter may be omitted. :meth:`check()` performs the separate final check that all mandatory
parameters are present:

>>> parameters = command.fill(count=10)
>>> command.check(parameters)

Error mitigation
================

The :attr:`apply_emt` attribute indicates whether error-mitigation techniques may expand and post-process a computation
using this command. It defaults to :code:`True`; custom commands whose results are not compatible with error mitigation
should set it to :code:`False`:

>>> command = pcvl.Command("custom_command", [], apply_emt=False)

Class reference
===============

.. autoclass:: perceval.runtime.command.Command
   :members:
