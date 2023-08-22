Basics
======

.. Notations and Definitions
   -------------------------

Perceval allows to simulate a wide range of optical devices and circuits.

Basic component Circuits
------------------------

Perceval allows to build photonic quantum circuits.  A *circuit* in
Perceval is an arrangement of devices -- such as beam splitters,
waveplates and so on -- or sub-circuits, with a fixed number :math:`m`
of spatial modes (which can be seen as "tracks" for photons).


These photonic quantum circuits are capable of performing a wide range of quantum computations.
For example, the well-known Boson Sampling proposal can be implemented by passing **n** single photons through a passive linear optical circuit designed to implement a so-called Haar random **m**-mode unitary transformation, then measuring the **m** output modes of this linear optical circuit.
On the other hand, **path encoding** each single photon by using two spatial modes per each photon essentially defines a qubit, the basic unit of quantum information processing. Thus, using **i** photons and **2i** modes, we can define an **i**-qubit input quantum state.
Passing this input state through the **m**-mode linear optical circuit, and using the remaining **n-i** single photons, and **m-2i**  remaining modes as ancillary modes and photons (to perform appropriate post-selections), one can implement any **i**-qubit unitary transformation on the input state, thereby allowing universal quantum computation. What we have just described is essentially  the famous Knill-Laflamme-Milburn protocol for universal qubit quantum computation. Alternatively, qubits can also be defined by performing a **polarization encoding** of the photons. Perceval supports both types of qubit encodings.


How to construct and simulate circuits is explained in more details in
section :ref:`Circuits`.


Sources
-------

While Perceval allows to simulate optical circuits alone by providing
an explicit input state -- i.e. specifying how many photons are sent
in each spatial mode -- it also allows to design *single photon
sources* to generate these photons.

Single photon sources are defined by specifying parameters such as
*brightness*, *purity* or *indistinguishability*.

Once a source has been defined, it can be plugged into a circuit, and
used to simulate the whole setup.



Encoding
--------

An optical circuit can be used to implement a quantum
circuit. Perceval does not impose any specific encoding for the
qu-bits. Many possible encodings exist, here we present two commonly
used encodings: *Spatial Modes encoding* and *Polarization Modes
encoding*. It is also however possible to conceive other types of
encodings.


Spatial Modes encoding
^^^^^^^^^^^^^^^^^^^^^^

In *Spatial Modes encoding* (also called *Path encoding*), each qubit
of a quantum circuit is encoded as a pair of spatial modes.  Each state of the
qubit corresponds to a Fock state where one photon is in one of the
spatial modes:

* qubit state :math:`|0\rangle` corresponds to having one photon in
  mode 0, and no photon in mode 1. Hence the corresponding Fock state
  is: :math:`|1,0\rangle`,
* similarly, qubit state :math:`|1\rangle` corresponds to having no photon in
  mode 0, and one photon in mode 1; the corresponding Fock state
  is: :math:`|0,1\rangle`.

.. note:: In spatial encoding, some Fock states don't correspond to any
   qubit state. An example of such a Fock state is :math:`|2,0\rangle`
   where two photons are sent in path 0 and no photon in path 1.

   More generally, any state which isn't a superposition of Fock
   states :math:`|0,1\rangle` and :math:`|1,0\rangle` can't be associated with a
   qubit state.



Polarization Modes encoding
^^^^^^^^^^^^^^^^^^^^^^^^^^^

In *Polarization Modes encoding*, each qubit is associated with one
spatial mode containing only one photon. A state of the qubit is
encoded using the polarization of the photon.

After choosing a basis for photon polarization, each state in the
qubit basis :math:`\{|0\rangle, |1\rangle\}` is associated with a
state in the polarization basis. For example, with polarization basis
:math:`\{|{P:H}\rangle, |{P:V}\rangle\}`:

* qubit state :math:`|0\rangle` corresponds to Fock state
  :math:`|{P:H}\rangle` -- which denotes the state where one photon is
  present in spatial mode 0 with *horizontal* polarization
* qubit state :math:`|1\rangle` corresponds to Fock state
  :math:`|{P:V}\rangle` -- which denotes the state where one photon is
  present in spatial mode 0 with *vertical* polarization


.. note:: In polarization encoding -- as in path encoding -- some Fock
   states can't be associated with a qubit state. In polarization
   encoding this is the case when either no photon or more than one
   photons are present in the spatial mode, or if the polarization is
   unspecified (see :ref:`Polarization`)


Simulation and Sampling
-----------------------

Consider a quantum circuit *C* which, when measured, outputs bitstrings *x* with a probability *p_x*. A classical algorithm providing a strong simulation of *C* is an algorithm which can **approximate**, up to very good precision (or more precisely *relative error*),  *p_x*   for all *x*. On the other hand, a classical algorithm that weakly simulates a quantum circuit is only required to **sample** outputs *x* with probabilities close to *p_x* (or in more precise terms up to a small error in the *total variational distance* ). For generic quantum circuits *C*, both these tasks become quickly unfeasible as the size of these circuits increases.
The optimized classical algorithms for simulating linear optical quantum circuits which are at the heart of Perceval allow us to perform both weak and strong simulations of photonic quantum circuits, with sizes comparable to those of circuit currently being implemented on quantum hardware. Thus, Perceval is a powerful tool for both experimentalists and theorists wishing to explore the capabilities of current and near-term photonic quantum hardware.
Perceval allows for strong and weak simulation of tasks such as Boson Sampling, Quantum machine learning, variational quantum algorithms, as well as small instances of Shors algorithm and Grover's search !

Numeric and Symbolic computation
--------------------------------

One of the key feature built-in most of the Perceval module is the ability to produce numeric and symbolic computation.
Symbolic computation use the excellent `sympy <https://www.sympy.org/en/index.html>`_ library and enable, when working
on smaller dimension problem, to get
analytical solution of a problem. Selection of the feature is enabled with the ``use_symbolic`` boolean parameter
available on numerous object methods.
