Circuit rendering skins
^^^^^^^^^^^^^^^^^^^^^^^

When rendering a :ref:`Circuit`, an :ref:`Experiment` or a :ref:`Processor`, you can select a skin which will change how
the components are displayed.
It's also possible to hack an existing skin to fit your needs or even create a new one.

Skin code reference
===================

All skins follow the :code:`ASkin` interface:

.. autoclass:: perceval.rendering.circuit.abstract_skin.ASkin
   :members:
