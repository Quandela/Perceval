Unitary Components
^^^^^^^^^^^^^^^^^^

Overview
========

.. list-table::
   :header-rows: 1
   :width: 100%

   * - Name
     - Class Name
     - ``PhysSkin`` display style
     - ``SymbSkin`` display style
     - Unitary Matrix
   * - :ref:`Beam Splitter`
     - ``BS``
     - .. image:: ../../_static/library/phys/bs.png
     - .. image:: ../../_static/library/symb/bs.png
     - Depends on the convention, see :ref:`Beam Splitter`
   * - :ref:`Phase Shifter`
     - ``PS``
     - .. image:: ../../_static/library/phys/ps.png
     - .. image:: ../../_static/library/symb/ps.png
     - :math:`\left[\begin{matrix}e^{i \phi}\end{matrix}\right]`
   * - :ref:`Permutation`
     - ``PERM``
     - .. image:: ../../_static/library/phys/perm.png
     - .. image:: ../../_static/library/symb/perm.png
     - Example of a two mode permutation: :math:`\left[\begin{matrix}0 & 1\\1 & 0\end{matrix}\right]`
   * - :ref:`Wave Plate`
     - ``WP``
     - .. image:: ../../_static/library/phys/wp.png
     - .. image:: ../../_static/library/symb/wp.png
     - :math:`\left[\begin{matrix}i \sin{\left(\delta \right)} \cos{\left(2 \xi \right)} + \cos{\left(\delta \right)} & i \sin{\left(\delta \right)} \sin{\left(2 \xi \right)}\\i \sin{\left(\delta \right)} \sin{\left(2 \xi \right)} & - i \sin{\left(\delta \right)} \cos{\left(2 \xi \right)} + \cos{\left(\delta \right)}\end{matrix}\right]`
   * - :ref:`Polarising Beam Splitter`
     - ``PBS``
     - .. image:: ../../_static/library/phys/pbs.png
     - .. image:: ../../_static/library/symb/pbs.png
     - :math:`\left[\begin{matrix}0 & 0 & 1 & 0\\0 & 1 & 0 & 0\\1 & 0 & 0 & 0\\0 & 0 & 0 & 1\end{matrix}\right]`
   * - :ref:`Polarisation Rotator`
     - ``PR``
     - .. image:: ../../_static/library/phys/pr.png
     - .. image:: ../../_static/library/symb/pr.png
     - :math:`\left[\begin{matrix}\cos{\left(\delta \right)} & \sin{\left(\delta \right)}\\- \sin{\left(\delta \right)} & \cos{\left(\delta \right)}\end{matrix}\right]`

Beam Splitter
=============

Beam Splitter conventions
-------------------------

.. autoenum:: perceval.components.unitary_components.BSConvention
   :members:

Three specialised conventions are defined, with a single :math:`\theta` parameter, as follow:

.. list-table::
   :header-rows: 1
   :width: 100%

   * - Convention
     - Unitary matrix
     - Default value (:math:`\theta=\pi/2`)
     - Comment
   * - ``Rx``
     - :math:`\left[\begin{matrix}\cos{(\theta/2)} & i \sin{(\theta/2)}\\i \sin{(\theta/2)} & \cos{(\theta/2)}\end{matrix}\right]`
     - :math:`\left[\begin{matrix}1 & i\\i & 1\end{matrix}\right]`
     - Symmetrical, default convention
   * - ``Ry``
     - :math:`\left[\begin{matrix}\cos{(\theta/2)} & -\sin{(\theta/2)}\\ \sin{(\theta/2)} & \cos{(\theta/2)}\end{matrix}\right]`
     - :math:`\left[\begin{matrix}1 & -1\\1 & 1\end{matrix}\right]`
     - Real, non symmetrical
   * - ``H``
     - :math:`\left[\begin{matrix}\cos{(\theta/2)} & \sin{(\theta/2)}\\ \sin{(\theta/2)} & -\cos{(\theta/2)}\end{matrix}\right]`
     - :math:`\left[\begin{matrix}1 & 1\\1 & -1\end{matrix}\right]`
     - Hadamard gate, ``HH=I``, non symmetrical

Each convention also accepts up to four additional phases, mimicking a phase shifter on each mode connected to the beam
splitter. For instance, with the ``Rx`` convention, the unitary matrix is defined by:

:math:`\left[\begin{matrix}e^{i(\phi_{tl}+\phi_{tr})} \cos{\left(\theta/2 \right)} & i e^{i (\phi_{tr}+\phi_{bl})} \sin{\left(\theta/2 \right)}\\i e^{i \left(\phi_{tl} + \phi_{br}\right)} \sin{\left(\theta/2 \right)} & e^{i (\phi_{br}+\phi_{bl})} \cos{\left(\theta/2 \right)}\end{matrix}\right]`

It is thus parametrized by :math:`\theta`, :math:`\phi_{tl}`, :math:`\phi_{bl}`, :math:`\phi_{tr}` and
:math:`\phi_{br}` angles, making this beam splitter equivalent to:

.. image:: ../../_static/img/bs_rx_4_phases.png

Beam Splitter reflectivity
--------------------------

:math:`\theta` relates to the reflectivity and :math:`\phi` angles correspond to relative phases between modes.
Beam splitters exist as bulk, fibered and on-chip components.

The relationship between the reflectivity :math:`R` and :math:`\theta` is: :math:`cos {\left( \theta/2 \right)} = \sqrt{R}`.

To create a beam splitter object with a given reflectivity value:

>>> from perceval.components import BS
>>> R = 0.45
>>> beam_splitter = BS(BS.r_to_theta(R))

Beam Splitter code reference
----------------------------

.. autoclass:: perceval.components.unitary_components.BS
   :members:
   :inherited-members:
   :exclude-members: identify, is_composite, transfer_from

Phase Shifter
=============

.. autoclass:: perceval.components.unitary_components.PS
   :members:

Permutation
===========

A permutation swaps multiple consecutive spatial modes.

To create a permutation ``PERM`` sending :math:`\ket{0,1}` to :math:`\ket{1,0}` and vice-versa:

>>> from perceval.components import PERM
>>> permutation = PERM([1, 0])

More generally, one can define a permutation on an arbitrary number of modes.
The permutation is described by a list of integers from 0 to :math:`l-1`, where :math:`l` is the length of the list.
The :math:`k^{th}` spatial input mode is sent to the spatial output mode corresponding to the :math:`k` th value in the list.

For instance the following defines
a 4-mode permutation. It matches the first input path (index 0) with the third output path (value 2), the second input path with the fourth output path, the third input path, with the first output path, and the fourth input path with the second output path.

>>> import perceval as pcvl
>>> import perceval.components.unitary_components as comp
>>> c = comp.PERM([2, 3, 1, 0])
>>> pcvl.pdisplay(c)
>>> pcvl.pdisplay(c.compute_unitary(), output_format=pcvl.Format.LATEX)

.. list-table::

   * - .. image:: ../../_static/library/phys/perm-2310.png
           :width: 180
     - .. math::
            \left[\begin{matrix}0 & 0 & 0 & 1\\0 & 0 & 1 & 0\\1 & 0 & 0 & 0\\0 & 1 & 0 & 0\end{matrix}\right]

.. autoclass:: perceval.components.unitary_components.PERM
   :members:

Unitary
=======

.. autoclass:: perceval.components.unitary_components.Unitary
   :members:

Wave Plate
==========

.. autoclass:: perceval.components.unitary_components.WP
   :members:

.. autoclass:: perceval.components.unitary_components.HWP
   :members:

.. autoclass:: perceval.components.unitary_components.QWP
   :members:

Polarisation Rotator
====================

.. autoclass:: perceval.components.unitary_components.PR
   :members:

Polarising Beam Splitter
========================

.. autoclass:: perceval.components.unitary_components.PBS
   :members:

Barrier
=======

.. autoclass:: perceval.components.unitary_components.Barrier
   :members:
