Components
==========

Overview
--------

.. list-table::
   :header-rows: 1
   :width: 100%

   * - Name
     - Object Name
     - Representation ``phys`` library
     - Representation ``symb`` library
     - Unitary Matrix
   * - :ref:`Beam Splitter`
     - ``BS``
     - .. image:: _static/library/phys/bs.png
     - .. image:: _static/library/symb/bs.png
     - :math:`\left[\begin{matrix}e^{i \phi_{a}} \cos{\left(\theta \right)} & i e^{i \phi_{b}} \sin{\left(\theta \right)}\\i e^{i \left(\phi_{a} - \phi_{b} + \phi_{d}\right)} \sin{\left(\theta \right)} & e^{i \phi_{d}} \cos{\left(\theta \right)}\end{matrix}\right]`
   * - :ref:`Phase Shifter`
     - ``PS``
     - .. image:: _static/library/phys/ps.png
     - .. image:: _static/library/symb/ps.png
     - :math:`\left[\begin{matrix}e^{i \phi}\end{matrix}\right]`
   * - :ref:`Permutation`
     - ``PERM``
     - .. image:: _static/library/phys/perm.png
     - .. image:: _static/library/symb/perm.png
     - :math:`\left[\begin{matrix}0 & 1\\1 & 0\end{matrix}\right]`
   * - :ref:`Waveplate`
     - ``WP``
     - .. image:: _static/library/phys/wp.png
     - .. image:: _static/library/symb/wp.png
     - :math:`\left[\begin{matrix}i \sin{\left(\delta \right)} \cos{\left(2 \xi \right)} + \cos{\left(\delta \right)} & i \sin{\left(\delta \right)} \sin{\left(2 \xi \right)}\\i \sin{\left(\delta \right)} \sin{\left(2 \xi \right)} & - i \sin{\left(\delta \right)} \cos{\left(2 \xi \right)} + \cos{\left(\delta \right)}\end{matrix}\right]`
   * - :ref:`Polarizing Beam Splitter`
     - ``PBS``
     - .. image:: _static/library/phys/pbs.png
     - .. image:: _static/library/symb/pbs.png
     - :math:`\left[\begin{matrix}0 & 0 & 1 & 0\\0 & 1 & 0 & 0\\1 & 0 & 0 & 0\\0 & 0 & 0 & 1\end{matrix}\right]`
   * - :ref:`Polarization Rotator`
     - ``PR``
     - .. image:: _static/library/phys/pr.png
     - .. image:: _static/library/symb/pr.png
     - :math:`\left[\begin{matrix}\cos{\left(\delta \right)} & \sin{\left(\delta \right)}\\- \sin{\left(\delta \right)} & \cos{\left(\delta \right)}\end{matrix}\right]`
   * - :ref:`Time Delay`
     - ``DT``
     - .. image:: _static/library/phys/dt.png
     - .. image:: _static/library/symb/dt.png
     - `N/A`

Description
-----------

Beam Splitter
^^^^^^^^^^^^^

Beam splitters couple two spatial modes together, implementing the following unitary acting on :math:`\ket{1,0}` and :math:`\ket{0,1}`:

:math:`\left[\begin{matrix} \cos{\left(\theta \right)} & i e^{i \phi} \sin{\left(\theta \right)}\\i e^{-i \phi} \sin{\left(\theta \right)} &  \cos{\left(\theta \right)}\end{matrix}\right]`

and are parametrized usually by angles :math:`\theta` and :math:`\phi`, where :math:`\theta`
relates to the reflectivity and :math:`\phi` corresponds to a relative phase between the modes.
Beam splitters exist as bulk, fibered and on-chip components.

It is also possible to use :math:`R` parameter with the following relationship: :math:`cos \theta= \sqrt{1-R}`.

In the ``phys`` library the beam splitters are described by four parameters: :math:`\theta, \phi_a, \phi_b, \phi_c`,
where :math:`\theta` and :math:`\phi_b` correspond to the above :math:`\theta` and :math:`\phi`. :math:`\phi_a`
and :math:`\phi_c` are additional phases that can be observed in actual devices.
These can that can be achieved in practice with the simplified unitary (present in the ``symb`` library) by using phase
shifters at the input and output of the beamsplitter and thus are included for compactness directly into the component.

To create a beam splitter object from the ``phys`` library:

>>> import perceval.lib.phys as phys
>>> beam_splitter = phys.BS()

By default
``theta`` is :math:`\pi/4`,
``phi_a`` is :math:`0`,
``phi_b`` is :math:`3\pi/2`,
``phi_d`` is :math:`\pi`.
These values can be modified by using optional parameters when creating a ``BS`` object.

In the ``symb`` library:

>>> import perceval.lib.symb as symb
>>> beam_splitter = symb.BS()

Only parameters ``theta`` and ``phi`` can be specified when using the ``symb`` library.


Phase Shifter
^^^^^^^^^^^^^

A phase shifter adds a phase :math:`\phi` on a spatial mode, which corresponds to a Z rotation in the Bloch sphere.

The definition of a phase shifter uses the same (non-optional) parameter ``phi`` in both libraries ``symb`` and ``phys``.
To create a phase shifter ``PS`` object:

>>> import perceval.lib.phys as phys
>>> phase_shifter = phys.PS(sp.pi/2) # phi = pi/2

or:

>>> import perceval.lib.symb as symb
>>> phase_shifter = symb.PS(sp.pi/2)


Permutation
^^^^^^^^^^^

A permutation exchanges two spatial modes, sending :math:`\ket{0,1}` to :math:`\ket{1,0}` and vice-versa.

To create a permutation ``PERM`` object corresponding to the above example:

>>> import perceval.lib.symb as symb
>>> permutation = symb.PERM([1,0])

or:

>>> import perceval.lib.phys as phys
>>> permutation = phys.PERM([1,0])

More generally one can define Permutation on an arbitrary number of modes.
The permutation should be described by a list of integers from 0 to :math:`l-1`, where :math:`l` is the length of the list.
The :math:`k` th spatial input mode is sent to the spatial output mode corresponding to the :math:`k` th value in the list.

For instance the following defines
a 4-mode permutation. It matches the first input path (index 0) with the third output path (value 2), the second input path with the fourth output path, the third input path, with the first output path, and the fourth input path with the second output path.

>>> import perceval as pcvl
>>> c=phys.PERM([2,3,0,1])
>>> pcvl.pdisplay(c)
>>> pcvl.pdisplay(c.compute_unitary(), output_format=pcvl.Format.LATEX)

.. list-table::

   * - .. image:: _static/library/phys/perm-2310.png
           :width: 180
     - .. math::
            \left[\begin{matrix}0 & 0 & 1 & 0\\0 & 0 & 0 & 1\\0 & 1 & 0 & 0\\1 & 0 & 0 & 0\end{matrix}\right]

We can do exactly the same with the symb library.

>>> c=symb.PERM([2,3,0,1])
>>> pcvl.pdisplay(c)
>>> pcvl.pdisplay(c.compute_unitary(), output_format=pcvl.Format.LATEX)

.. list-table::

   * - .. image:: _static/library/symb/perm-2310.png
           :width: 180
     - .. math::
            \left[\begin{matrix}0 & 0 & 1 & 0\\0 & 0 & 0 & 1\\0 & 1 & 0 & 0\\1 & 0 & 0 & 0\end{matrix}\right]

Waveplate
^^^^^^^^^^

A waveplate acts on the polarisation modes of a single spatial mode. It is described by the following unitary:

.. math::
    \left[\begin{matrix}i \sin{\left(\delta \right)} \cos{\left(2 \xi \right)} + \cos{\left(\delta \right)} & i \sin{\left(\delta \right)} \sin{\left(2 \xi \right)}\\i \sin{\left(\delta \right)} \sin{\left(2 \xi \right)} & - i \sin{\left(\delta \right)} \cos{\left(2 \xi \right)} + \cos{\left(\delta \right)}\end{matrix}\right]

:math:`\delta` is a parameter proportional to the thickness of the waveplate and :math:`\xi` represents the angle of the waveplate's optical axis in the :math:`\left\{\ket{H}, \ket{V}\right\}` plane. Especially important is the case that :math:`\delta=\pi/2`, known as a half-wave plate, which rotates linear polarisations in the :math:`\left\{\ket{H}, \ket{V}\right\}` plane.

Polarizing Beam Splitter
^^^^^^^^^^^^^^^^^^^^^^^^

A polarising beam splitter converts a superposition of polarisation modes in a single spatial mode to the corresponding equal-polarisation superposition of two spatial modes,and vice versa, and so in this sense allow us to translate between polarisation and spatial modes. The unitary matrix associated to a polarising beam splitter acting on the tensor product of the spatial mode and the polarisation mode is:

.. math::
    \left[\begin{matrix}0 & 0 & 1 & 0\\0 & 1 & 0 & 0\\1 & 0 & 0 & 0\\0 & 0 & 0 & 1\end{matrix}\right]

Polarization Rotator
^^^^^^^^^^^^^^^^^^^^

A polarization rotator is an optical device that rotates the polarization axis of a linearly polarized light beam by an angle of choice.
Such devices can be based on the Faraday effect, on birefringence, or on total internal reflection.
Rotators of linearly polarized light have found widespread applications in modern optics since laser beams tend to be linearly polarized and it is often necessary to rotate the original polarization to its orthogonal alternative.

See https://en.wikipedia.org/wiki/Polarization_rotator for more details.

Time Delay
^^^^^^^^^^

Time Delay is a special component corresponding to a roll of optical fiber making as an effect to delay a photon.
Parameter of the Time Delay is the fraction of a period the delay should be.

For instance ``DT(0.5)`` will make a delay on the line corresponding to half of a period.
