Non-unitary Components
^^^^^^^^^^^^^^^^^^^^^^

Supported non-unitary components
================================

.. list-table::
   :header-rows: 1
   :width: 100%

   * - Name
     - Class Name
     - ``PhysSkin`` display style
     - ``SymbSkin`` display style
   * - :ref:`Time Delay`
     - ``TD``
     - .. image:: ../../_static/library/phys/dt.png
     - .. image:: ../../_static/library/symb/dt.png
   * - :ref:`Loss Channel`
     - ``LC``
     - .. image:: ../../_static/library/phys/lc.png
     - .. image:: ../../_static/library/symb/lc.png

Time Delay
==========

.. autoclass:: perceval.components.non_unitary_components.TD
   :members:
   :inherited-members:
   :exclude-members: identify, is_composite, transfer_from

Loss Channel
============

.. autoclass:: perceval.components.non_unitary_components.LC
   :members:
   :inherited-members:
   :exclude-members: identify, is_composite, transfer_from
