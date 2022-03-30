Polarization
============

Polarization encoding is stored in :ref:`Annotated Basic State` objects as a special ``P`` annotation.

Their value follows `Jones calculus <https://en.wikipedia.org/wiki/Jones_calculus>`_. Annotations values are represented
by two angles :math:`(\theta, \phi)`.

The representation of the polarization in :math:`\begin{pmatrix}E_h\\E_v\end{pmatrix}` basis is obtained by applying
Jones conversion: :math:`\overrightarrow{J}=\begin{pmatrix}\cos \frac{\theta}{2}\\e^{i\phi}\sin \frac{\theta}{2}\end{pmatrix}`. The same
can also be noted: :math:`\cos \frac{\theta}{2}\ket{H}+e^{i\phi}\sin \frac{\theta}{2}\ket{V}`.

For instance, the following defines a polarization with :math:`\theta=\frac{\pi}{2},\phi=\frac{\pi}{4}` corresponding to
Jones vector: :math:`\begin{pmatrix}\cos \frac{\pi}{4}\\e^{i\frac{\pi}{4}}\sin \frac{\pi}{4}\end{pmatrix}`

.. code-block:: python

    >>> p = pcvl.Polarization(sp.pi/2, sp.pi/4)
    >>> p.project_ev_eh()
    (sqrt(2)/2, sqrt(2)*exp(I*pi/4)/2)

It is also possible to use ``H``, ``V``, ``D``, ``A``, ``L`` and ``R`` as shortcuts to predefined values:

.. list-table::
   :header-rows: 1

   * - Code
     - :math:`(\phi,\theta)`
     - Jones vector
   * - ``H``
     - :math:`(0,0)`
     - :math:`\begin{pmatrix}1\\0\end{pmatrix}`
   * - ``V``
     - :math:`(\pi,0)`
     - :math:`\begin{pmatrix}0\\1\end{pmatrix}`
   * - ``D``
     - :math:`(\frac{\pi}{2},0)`
     - :math:`\frac{1}{\sqrt 2}\begin{pmatrix}1\\1\end{pmatrix}`
   * - ``A``
     - :math:`(\frac{\pi}{2},\pi)`
     - :math:`\frac{1}{\sqrt 2}\begin{pmatrix}1\\-1\end{pmatrix}`
   * - ``L``
     - :math:`(\frac{\pi}{2},\frac{\pi}{2})`
     - :math:`\frac{1}{\sqrt 2}\begin{pmatrix}1\\i\end{pmatrix}`
   * - ``R``
     - :math:`(\frac{\pi}{2},\frac{3\pi}{2})`
     - :math:`\frac{1}{\sqrt 2}\begin{pmatrix}1\\-i\end{pmatrix}`

.. code-block:: python

    >>> p = pcvl.Polarization("D")
    >>> p.theta_phi
    (pi/2, 0)
    >>> p.project_ev_eh())
    (sqrt(2)/2, sqrt(2)/2)

Defining states with polarization is then simply to use the :ref:`Annotation` ``P``:

.. code-block:: python

    >>> st2 = pcvl.AnnotatedBasicState("|{P:H},{P:V}>")
    >>> st2 = pcvl.AnnotatedBasicState("|{P:(sp.pi/2,sp.pi/3)>")

If polarization is used for any photon in the state, the state is considered as using polarization:

.. code-block:: python

    >>> pcvl.AnnotatedBasicState("|{P:H},0,{P:V}>").has_polarization
    True
    >>> pcvl.AnnotatedBasicState("|{P:V},0,1>").has_polarization
    True
    >>> pcvl.AnnotatedBasicState("|1,0,1>").has_polarization
    False

.. note::
   To simplify the notation:

   * linear polarization can be defined with a single parameter: ``{P:sp.pi/2}`` is equivalent to ``{P:(sp.pi/2,0}``

   * if the polarization annotation is omitted for some photons, these photons will be considered as having a horizontal polarization.


.. code-block: python

   >>> st3 = pcvl

See :ref:`Polarization Object` code documentation.