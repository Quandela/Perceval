DisplayConfig
=============

This class allows you to choose a display skin for pdisplay and save the chosen skin across perceval runs.

.. code-block::

  from perceval.rendering import DisplayConfig, SymbSkin
  DisplayConfig.select_skin(SymbSkin) # SymbSkin will be used by default by pdisplay if no other skin is defined.
  DisplayConfig.save() # Will save the current DisplayConfig into your Perceval persistent configuration.

.. autoclass:: perceval.rendering.DisplayConfig
   :members:
   :inherited-members:
