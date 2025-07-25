Circuit
=======

The basic usage and definition of Circuit can be found in the dedicated section :ref:`Circuits`.

.. warning::
  Move everything that is not essential from :ref:`Circuits` to here.

Accessing the components
^^^^^^^^^^^^^^^^^^^^^^^^

Circuits can be iterated over to retrieve their components.
Each component is a tuple :code:`(r, c)`
where :code:`r` is a tuple of integers corresponding to the modes of the component, in ascending order,
and :code:`c` is the component instance itself.

>>> import perceval as pcvl
>>> circuit = pcvl.Circuit(3) // pcvl.BS() // (1, pcvl.PS(1)) // (1, pcvl.BS())
>>> for r, c in circuit:
>>>     print(r, c.name)
(0, 1) BS.Rx
(1,) PS
(1, 2) BS.Rx

.. note::
  The iterator on a :code:`Circuit` flattens the circuit structure,
  so only basic components will be returned when using a :code:`for` loop on a circuit.

It is also possible to access directly a component from a circuit using `row` and `column` indices - note that a same
component may have different column indices for the different rows it spans over:

>>> c = Circuit(2) // comp.BS.H() // comp.PS(P("phi1")) // comp.BS.Rx() // comp.PS(P("phi2"))
>>> print(c[1, 1].describe())
BS(convention=BSConvention.Rx)
>>> print(c[0, 2].describe())
BS(convention=BSConvention.Rx)

Circuit and parameters
^^^^^^^^^^^^^^^^^^^^^^

For circuits or components using symbolic parameters (see :ref:`Parameter`), some convenient ways to access them exist.
Note that two parameters in the same circuit can't have the same name.
If a parameter's name appears in more than one component, it can only be the same :code:`Parameter` instance.

Probably the most useful of them is :meth:`assign()`,
as it allows setting the value for all variable parameters of a circuit at once,
even without having to store them somewhere.

>>> p = pcvl.P("phi0")
>>> c = pcvl.PS(p)
>>> c.assign({"phi0": 2.53})
>>> print(float(p))
2.53

.. note::
  The :code:`assign` argument of :meth:`compute_unitary()` does exactly that if you want to compute the unitary with values.
  Beware however that this has the side-effect of changing the values of the parameters even outside :meth:`compute_unitary()`.
  You can remove the values for the variable parameters to get back sympy expressions using the circuit's :meth:`reset_parameters()` method.

The names of the parameters can be obtained using the :meth:`params` property.
Note that this includes the fixed parameters if there are any.

To get the :code:`Parameter` itself, there are three ways:

- use the :meth:`param("param name")` method to retrieve a single parameter from its name.
- use the :meth:`get_parameters()` method that gives all the parameters defined by the arguments
  (variable or all, with or without expressions). This is the preferred method for getting all parameters.

    >>> c = BS(theta=pcvl.P("alpha1")) // PS(pcvl.P("phi")) // BS(theta=pcvl.P("alpha2"))
    >>> for params in c.get_parameters():
    >>>     print(param)
    Parameter(name='alpha1', value=None, min_v=0.0, max_v=12.566370614359172)
    Parameter(name='phi', value=None, min_v=0.0, max_v=6.283185307179586)
    Parameter(name='alpha2', value=None, min_v=0.0, max_v=12.566370614359172)

- use the :meth:`vars` property to get a dictionary mapping the name of the variable parameters and their instances.

.. autoclass:: perceval.components.linear_circuit.Circuit
   :members:
   :inherited-members:
   :special-members: __ifloordiv__, __floordiv__, __matmul__, __imatmul__, __iter__
