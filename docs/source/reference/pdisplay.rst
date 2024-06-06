PDisplay - Pretty Display
=========================

>>> import perceval as pcvl
>>> processor = pcvl.catalog['heralded cz'].build_processor()
>>> pcvl.pdisplay(processor, output_format=pcvl.Format.TEXT)

.. code-block::

            ╔[Heralded CZ]╗
            ║░░░░░░░░░░░░░║
      (]────╫░░░░░░░░░░░░░╫──[)
      ctrl  ║░░░░░░░░░░░░░║[ctrl]
            ║░░░░░░░░░░░░░║
      (]────╫░░░░░░░░░░░░░╫──[)
      ctrl  ║░░░░░░░░░░░░░║[ctrl]
            ║░░░░░░░░░░░░░║
      (]────╫░░░░░░░░░░░░░╫──[)
      data  ║░░░░░░░░░░░░░║[data]
            ║░░░░░░░░░░░░░║
      (]────╫░░░░░░░░░░░░░╫──[)
      data  ║░░░░░░░░░░░░░║[data]
            ║░░░░░░░░░░░░░║
   (1]──────╫░░░░░░░░░░░░░╫──[1)
   herald0  ║░░░░░░░░░░░░░║[herald0]
            ║░░░░░░░░░░░░░║
   (1]──────╫░░░░░░░░░░░░░╫──[1)
   herald1  ║░░░░░░░░░░░░░║[herald1]
            ╚             ╝


.. automodule:: perceval.rendering.pdisplay
   :members:
