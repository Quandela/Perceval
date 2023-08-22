Circuits
========


Circuits Overview
-----------------

The concept of *circuit* is central in Perceval and the key component of the instrumentalist and the theoretician
building a photonic quantum device. Perceval is giving concrete tools to reproduce an optical set-up in an optic
laboratory with beams, mirrors and physical component but also photonic chips like a generic interferometer.

.. list-table::
   :width: 100%

   * - .. figure:: _static/img/grover-circuit.png
         :align: center
         :width: 500

         Optimized Grover algorithm as proposed in :cite:p:`kwiat_grovers_2000`.
     - .. figure:: _static/img/grover-perceval.png
         :align: center
         :width: 500

         The equivalent circuit in Perceval

   * - .. figure:: _static/img/quandela-4-mode-chip-circuit.png
         :align: center

         A 4-mode photonic chip (*Copyright Quandela 2022*)
     - .. figure:: _static/img/quandela-4-mode-chip-circuit.png
         :align: center
         :height: 319

         The equivalent circuit in Perceval

.. kept as comment

   * - .. figure:: _static/img/grover-circuit.png
         :align: center
         :width: 500

         Optimized Grover algorithm proposed in [Kwiat2000]_.
     - **TODO**: The equivalent PHYS circuit

What is a Circuit ?
-------------------

In Perceval a *circuit* represents a setup of optical components, used
to guide and act on photons.

A circuit has a fixed number of *spatial modes* (sometimes also called
*paths* or *ports*) :math:`m`, which is the same for input as for output
spatial modes.

Simple examples of circuits are common optical devices such as beam
splitters, phase shifters, or wave plates. Perceval provides a
collection of these (see :ref:`Components`).

.. figure:: _static/library/phys/bs.png
   :scale: 50 %
   :align: center

   A beam splitter as a circuit in Perceval.

In particular, note that:

* *sources* aren't circuits, since they do not have input spatial
  modes (they don't guide or act on incoming photons, but *produce*
  photons that are sent into a circuit),
* *detectors* aren't circuits either, for similar reasons


.. warning::
   An *optical circuit* (just called
   "circuit" here) isn't the same as a *quantum circuit*. Quantum
   circuits act on *qubits*, i.e. abstract systems in a 2-dimentional
   Hilbert space; while optical circuits act on *photons*
   distributed in spatial modes. It is possible to simply encode
   qubits with photons in an optical circuit; some encodings are
   presented in the :ref:`Basics` section.

.. figure:: _static/img/basic-circuit.png
  :scale: 50 %
  :align: center

  A simple circuit with 4 spatial modes, containing a beam splitter
  (being itself a circuit) between second and third spatial modes

Circuits can be *combined* together and used as building blocks to
construct larger circuits. In the example above, a beam splitter,
which is a builtin circuit with 2 spatial modes, has been used in a
larger circuit with 4 spatial modes.

The lines, corresponding to spatial modes, are representing optical
fibers on which photons are sent from the left to the right.

.. The way these lines are used to represent
   logical modes are up to the circuit design.


Building a Circuit
------------------

Each circuit corresponds to a ``Circuit`` object.
To instantiate a circuit, simply pass the number of modes as an argument:

>>> # create a new circuit with 3 spatial modes
>>> my_circuit = pcvl.Circuit(3)



.. warning::

  Ports are using `0-based numbering <https://en.wikipedia.org/wiki/Zero-based_numbering>`_ - so port 0 is
  corresponding to the first line, ... port :math:`(m-1)` is corresponding to the :math:`m`-th line.


Circuit library
^^^^^^^^^^^^^^^

Perceval provides a library of predefined components, located in ``perceval.components``:

>>> import perceval.components as comp

This library contain simple circuits stored in a few sub-packages. For instance:

* ``unitary_components`` provides circuits which can be represented by a unitary matrix.
* ``non_unitary_components`` provides other types of circuit, such as time delays.

Circuit Rendering
^^^^^^^^^^^^^^^^^

Perceval provides ``pdisplay`` function as an easy and ergonomic way to display a lot of Perceval objects, given the
execution context (Jupyter notebook, IDE, command line script). Circuit rendering is built upon a skin system allowing
to style your output image.

Perceval comes with two built-in skins:

* ``SymbSkin``: a sober black and white skin
* ``PhysSkin``: a more colorful "real-life" skin

The following figure shows the same Mach Zehnder Interferometer circuit represented with the ``SymbSkin`` on the left
and the ``PhysSkin`` on the right:

.. figure:: _static/img/comparing-symb-and-phys.png
  :align: center
  :width: 75%

>>> from perceval.rendering.circuit import SymbSkin, PhysSkin
>>> perceval.pdisplay(circuit, skin=SymbSkin())

See :ref:`Components` for an overview of the circuits provided by
Perceval.

.. Elementary Circuits

   The simplest circuits are the predefined :ref:`Components` that are included in Perceval.

   To define a circuit, you simply need to instantiate the corresponding object:

   >>> bs = comp.BS()

.. And you can use it the same way you will use a complex circuit as described in :ref:`Complex Circuits`.

In addition to the generic properties of complex circuits, elementary
circuits also have the following
features:

* **Definition**: ``definition()`` method gives the definition of the circuit - allowing to know in particular, the parameters to use
  and the way the unitary matrix is computed.

  >>> pcvl.pdisplay(comp.BS().definition())
  ⎡cos(theta)               I*exp(-I*phi)*sin(theta)⎤
  ⎣I*exp(I*phi)*sin(theta)  cos(theta)              ⎦

* **Parameterization**: most of the elementary circuits are defined by parameters. For instance a phase shifter will take
  the phase :math:`\phi` as a parameter. You can either initialize these parameters with a fixed numeric value:
  ``comp.PS(1.57)``, with a symbolic value ``comp.PS(sp.pi/2)`` or with a non fixed parameter:
  ``comp.PS(pcvl.P("alpha"))``. See :ref:`Parameters` for more information.

Defining circuits from a unitary matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can also define any circuit directly from a unitary matrix.  In
Perceval a unitary matrix corresponds to a ``Matrix`` object, which
can simply created from a list matrix. The code:

>>> M = pcvl.Matrix([[0, 0, 1],
...                  [0, 1, 0],
...                  [1, 0, 0]])


corresponds to the unitary matrix: :math:`\left[\begin{matrix}0 & 0 &
1\\0 & 1 & 0\\1 & 0 & 0\end{matrix}\right]`

The following then defines a circuit corresponding to this matrix:

>>> c1 = comp.Unitary(U=M)


You might also want to decompose the unitary matrix into a physical circuit using decomposition elements.
Let us define for instance:

>>> ub = Circuit(2, name="ub") // comp.BS() // (0, comp.PS(phi=pcvl.Parameter("φ_a"))) // comp.BS() // (1, comp.PS(phi=pcvl.Parameter("φ_b")))

Then you can build a circuit using the method :meth:`perceval.components.circuit.Circuit.decomposition`:

>>> c2 = pcvl.Circuit.decomposition(M, mzi, shape="triangle")
>>> c2.describe()
Circuit(3).add((0, 1), phys.BS()).add(0, phys.PS(phi=pi)).add((0, 1), phys.BS()).add(1, phys.PS(phi=0.681255)).add((1, 2), phys.BS()).add(1, phys.PS(phi=-pi)).add((1, 2), phys.BS()).add(2, phys.PS(phi=0.124498)).add((0, 1), phys.BS()).add(0, phys.PS(phi=pi)).add((0, 1), phys.BS()).add(1, phys.PS(phi=3.974189))
>>> pcvl.pdisplay(c2)

.. image:: _static/img/decompose-matrix.png

Some additional parameters can simplifiy the decomposition:

* `permutation`: if set to a permutation component, permutations will be used when possible instead of a unitary block

>>> import perceval as pcvl
>>> import perceval.components.unitary_components as comp
>>> C1 = pcvl.Circuit.decomposition(comp.PERM([3, 2, 1, 0]).compute_unitary(False),
>>>                                 comp.BS(theta=pcvl.Parameter("theta")),
>>>                                 permutation=comp.PERM,
>>>                                 shape="triangle")
>>>pcvl.pdisplay(C1)

.. image:: _static/img/permutations-perm.png
   :width: 8cm

* `constraints`: you can provide a list of constraints on the different parameters of the unitary blocks to try to find
  circuits with constrained parameters. Each constraint is a t-uple of `None` or numerical value. When decomposing the
  circuit, the parameters will be searched iteratively in the constrained spaces. For instance: `[(0, None), (np.pi/2, None), (None, None)]`
  will allow to look for parameters pairs where the first parameter is 0 or :math:`pi/2`, or any value if
  no solution is found with the first constraints.

>>> U=1/3*np.array([[np.sqrt(3),-np.sqrt(6)*1j,0,0,0,0],
>>>                 [-np.sqrt(6)*1j,np.sqrt(3),0,0,0,0],
>>>                 [0,0,np.sqrt(3),-np.sqrt(3)*1j,-np.sqrt(3)*1j,0],
>>>                 [0,0,-np.sqrt(3)*1j,np.sqrt(3),0,np.sqrt(3)],
>>>                 [0,0,-np.sqrt(3)*1j,0,np.sqrt(3),-np.sqrt(3)],
>>>                 [0,0,0,np.sqrt(3),-np.sqrt(3),-np.sqrt(3)]])
>>> ub = comp.BS(theta=pcvl.P("theta")) // comp.PS(phi=pcvl.P("phi"))
>>> C1 = pcvl.Circuit.decomposition(U,
>>>                                 ub,
>>>                                 shape="triangle", constraints=[(None,0),(None,np.pi/2),
>>>                                                                (None,3*np.pi/2),(None,None)])

.. image:: _static/img/cnot-decomposed.png
   :width: 8cm

* `phase_shifter_fn`: if you provide a phase-shifter to this parameter, the decomposition will add a layer of phases
  making the decomposed circuit strictly equivalent to the initial unitary matrix. In most cases, you can however omit
  this layer.

* finally, you can also pass simpler unitary blocks - for instance a simple beamsplitter without phase, however in these
  cases, you might not obtain any solution in the decomposition

Accessing components in a circuit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is possible to access directly a component from a circuit using `row` and `column` indices - note that a same
component may have different column indices for the different rows it spans over:

>>> c = Circuit(2) // comp.BS.H() // comp.PS(P("phi1")) // comp.BS.Rx() // comp.PS(P("phi2"))
>>> print (c[1, 1].describe())
BS(convention=BSConvention.Rx)
>>> print (c[0, 2].describe())
BS(convention=BSConvention.Rx)

Circuit simplification
^^^^^^^^^^^^^^^^^^^^^^

The syntax is the following:

>>> from perceval.utils.algorithms.simplification import simplify
>>>
>>> c = Circuit() // ...
>>> simplified_c = simplify(c, display = False)

Circuit simplification takes a circuit and does the following:

* For phase shifters, add their phase if they are not parameters and combine them into a single phase shifters (work through permutations). If :code:`display == False`, removes them if their added phase is :math:`0` or :math:`2\pi`.
* For Permutations, if two permutations are consecutive, they are combined into a single permutation. For single permutations, fixed modes at the extremities are removed. If they are not just consecutive, try to compute a "better" permutation, then if it is better, moves the components accordingly to this new permutation. Display changes how a permutation is evaluated.


Complex Circuits
^^^^^^^^^^^^^^^^

Assembling circuits
"""""""""""""""""""

A circuit is defined by using :ref:`Circuit` object as following:

>>> c = pcvl.Circuit(m)

Where ``m`` is the number of modes of the circuit. See :ref:`Circuit` for additional parameters.

Then components of the circuit are added with the ``add`` primitive:

>>> c.add((0, 1), comp.BS())

Where:

* The first parameter is either the *port range* (here ports 0 and 1), or the upper where the component should be added.
  The previous declaration is equivalent to:

  >>> c.add(0, comp.BS())

* The second parameter is a circuit, it can be an elementary circuit as in our example, or another complex circuit.

.. tip::

  It is possible to add multiple components in a single statement allowing for simpler circuit declaration:

  >>> mzi = pcvl.Circuit(2).add(0, comp.BS()).add(0, comp.PS(pcvl.Parameter("phi1")))\
  ...                      .add(0, comp.BS()).add(0, comp.PS(pcvl.Parameter("phi2"))))

  alternatively, you can also use ``//`` notation for more compact definition, and start from unitary circuit:

  >>> mzi = comp.BS() // (0, comp.PS(pcvl.Parameter("phi1"))) // comp.BS() // (0, comp.PS(pcvl.Parameter("phi2")))

Generic Interferometer
""""""""""""""""""""""

It is also possible to define generic interferometers with the class
:meth:`perceval.components.GenericInterferometer`.

For instance the following defines a triangular interferometer on 8 modes using a beam splitter
and a phase shifter as base components:

>>> c = pcvl.GenericInterferometer(8,
...                                lambda i: comp.BS() // comp.PS(pcvl.P("φ%d" % i)),
...                                shape="triangle")
>>> pcvl.pdisplay(c)

.. figure:: _static/img/generic-interferometer.png
  :align: center
  :width: 75%

Sub-circuits
^^^^^^^^^^^^

When you assemble to build a circuit, you can naturally include a complex circuit into another one:

>>> bsps = comp.BS().add(0, comp.PS(sp.pi/2))
>>> c = pcvl.Circuit(3).add(0, bsps).add(1, bsps)
>>> pcvl.pdisplay(c)

.. image:: _static/img/wmerge.png
   :width: 300px

By default, the new circuits merge all the circuits into a single one by using ``merge=False`` parameter:

>>> c = pcvl.Circuit(3).add(0, bsps).add(1, bsps, merge=False)
>>> pcvl.pdisplay(c)

.. image:: _static/img/wnomerge.png
   :width: 300px

This can be particularly useful to get a better visual organization of large circuit.

Unitary Matrices
----------------

Except for circuits using :ref:`Time Delay`, any circuit can be converted into its unitary matrix. Depending if your
circuit is using :ref:`Parameters` or not, the unitary matrix will be symbolic or numeric.

>>> chip4mode = pcvl.Circuit(m=4, name="QChip")
>>> phis = [pcvl.Parameter("phi1"), pcvl.Parameter("phi2"),
...         pcvl.Parameter("phi3"), pcvl.Parameter("phi4")]
>>> (chip4mode
...  .add((0, 1), comp.BS()).add((2, 3), comp.BS()).add((1, 2), comp.PERM([1, 0]))
...  .add(0, comp.PS(phis[0])).add(2, comp.PS(phis[2])).add((0, 1), comp.BS())
...  .add((2, 3), comp.BS()).add(0, comp.PS(phis[1])).add(2, comp.PS(phis[3]))
...  .add((0, 1), comp.BS()).add((2, 3), comp.BS()))
>>> pcvl.pdisplay(chip4mode.U)

:math:`\left[\begin{matrix}\frac{\sqrt{2} \left(- e^{i \phi_{1}} + e^{i \left(\phi_{1} + \phi_{2}\right)}\right)}{4} & \frac{\sqrt{2} i \left(- e^{i \phi_{1}} + e^{i \left(\phi_{1} + \phi_{2}\right)}\right)}{4} & \frac{\sqrt{2} i \left(e^{i \phi_{2}} + 1\right)}{4} & - \frac{\sqrt{2} \left(e^{i \phi_{2}} + 1\right)}{4}\\\frac{\sqrt{2} i \left(e^{i \phi_{1}} + e^{i \left(\phi_{1} + \phi_{2}\right)}\right)}{4} & - \frac{\sqrt{2} \left(e^{i \phi_{1}} + e^{i \left(\phi_{1} + \phi_{2}\right)}\right)}{4} & \frac{\sqrt{2} \left(1 - e^{i \phi_{2}}\right)}{4} & \frac{\sqrt{2} i \left(1 - e^{i \phi_{2}}\right)}{4}\\\frac{\sqrt{2} i \left(- e^{i \phi_{3}} + e^{i \left(\phi_{3} + \phi_{4}\right)}\right)}{4} & \frac{\sqrt{2} \left(- e^{i \phi_{3}} + e^{i \left(\phi_{3} + \phi_{4}\right)}\right)}{4} & - \frac{\sqrt{2} \left(e^{i \phi_{4}} + 1\right)}{4} & \frac{\sqrt{2} i \left(e^{i \phi_{4}} + 1\right)}{4}\\- \frac{\sqrt{2} \left(e^{i \phi_{3}} + e^{i \left(\phi_{3} + \phi_{4}\right)}\right)}{4} & \frac{\sqrt{2} i \left(e^{i \phi_{3}} + e^{i \left(\phi_{3} + \phi_{4}\right)}\right)}{4} & \frac{\sqrt{2} i \left(1 - e^{i \phi_{4}}\right)}{4} & \frac{\sqrt{2} \left(1 - e^{i \phi_{4}}\right)}{4}\end{matrix}\right]`

See :meth:`perceval.components.circuit.Circuit.compute_unitary` for more information.

Circuit Rewriting
-----------------

To enable circuit rewriting operations introduced in :cite:p:`clement2022lov`, the following methods are available for
matching a specific pattern in a circuit, and to replace the corresponding sub-circuit by an equivalent circuit.

The complete sequence is the following:

>>> while True:
...    # identify one instance of the parameterized "pattern" within a circuit
...    matched = circuit.match(pattern, browse=True)
...    # check if an occurence was found
...    if matched is None:
...       break
...    # transform the list of matched components into a sub-circuit
...    idx = a.isolate(list(matched.pos_map.keys()))
...    # check if an equivalent "rewrite" circuit can be found
...    res = optimize(rewrite, v, frobenius, sign=-1)
...    # check if we can rewrite this pattern closely enough to original circuit, here with frobenius distance
...    if res.fun > 1e-6:
...        break
...    # replace the subcircuit by a copy of the pattern
...    a.replace(idx, rewrite.copy(), merge=True)
...    # reset all parameters of pattern and rewrite circuit that have been instantiated by
...    # the matching/optimize process
...    pattern.reset_parameters()
...    rewrite.reset_parameters()

See *Reference to Notebook* for a complete functional example.
