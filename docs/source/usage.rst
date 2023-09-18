Usage
=====

Installation
------------

To use perceval, you can directly install the pip wheel in a virtual environment:

.. code-block:: bash

   (.venv) $ pip install perceval-quandela

Alternatively, if you are interested in contributing to the project - you can clone the project from github:

.. code-block:: bash

   (venv) $ git clone https://github.com/quandela/Perceval
   (venv) $ cd Perceval
   (venv) $ python setup.py install # or develop for developers

At this point you can directly use Perceval. To follow the tutorial below, we do recommend running the code in a
jupyter notebook, or in the Python console of your favorite IDE (Spyder, Pycharm) but you can also run it in a
terminal python console.


First Circuit
-------------

To begin, let's import the library, the unitary components and the simulator

>>> import perceval as pcvl
>>> import perceval.components.unitary_components as comp
>>> from perceval.simulators import Simulator

As a first circuit, you can try to display a single component, like a beam splitter for exemple.

>>> component = comp.BS()
>>> pcvl.pdisplay(component)

You can also access really easily to different aspects of the component, such as the number of modes of your circuit,

>>> components.m
2

or the matrix definition of it

>>> pcvl.pdisplay(component.definition())
⎡exp(I*(phi_tl + phi_tr))*cos(theta/2)    I*exp(I*(phi_bl + phi_tr))*sin(theta/2)⎤
⎣I*exp(I*(phi_br + phi_tl))*sin(theta/2)  exp(I*(phi_bl + phi_br))*cos(theta/2)  ⎦

The parameters above (here with default values) are accessible with

>>> components.get_parameters(all_params=True)
[Parameter(name='theta', value=pi/2, min_v=0.0, max_v=12.566370614359172),
Parameter(name='phi_tl', value=0, min_v=0.0, max_v=6.283185307179586),
Parameter(name='phi_bl', value=0, min_v=0.0, max_v=6.283185307179586),
Parameter(name='phi_tr', value=0, min_v=0.0, max_v=6.283185307179586),
Parameter(name='phi_br', value=0, min_v=0.0, max_v=6.283185307179586)]

You can also check if the matrix is unitary

>>> component.U.is_unitary()
True

Simulation
----------

The following part will allow you to simulate the behavior of photons crossing the circuit

First of all, let's choose an input state for our simulation. The following line instanciate an input state that represent one photon going in the second mode. And you can check the number of photons of your input state

>>> input_state = pcvl.BasicState("|0,1>")
>>> input_state.n
1

Then you have to choose a backend for the simulation and create a Simulator

>>> backend = pcvl.BackendFactory().get_backend()
>>> simulator = Simulator(backend)
>>> simulator.set_circuit(component)

Now you can get the state vector at the output of your component:

>>> print(simulator.evolve(input_state))
sqrt(2)*I/2*|1,0>+sqrt(2)/2*|0,1>

You can also get output samples by using the backend of your choice (here the Naive one)

>>> p = pcvl.Processor("Naive", component)
>>> p.with_input(input_state)
>>> sampler = pcvl.algorithm.Sampler(p)
>>> samples = sampler.samples(10)
>>> for state in samples['results']:
...    print(state)
...
|0,1>
|0,1>
|0,1>
|1,0>
|1,0>
|0,1>
|0,1>
|0,1>
|0,1>
|0,1>


You can also get the full probability distribution for any input state.

>>> distrib = pcvl.algorithm.Analyzer(p,
...                           [pcvl.BasicState([0, 1]),
...                             pcvl.BasicState([1, 0]),
...                             pcvl.BasicState([1, 1])],)
>>> pcvl.pdisplay(ca)
+-------+-------+-------+-------+-------+-------+
|       | |1,0> | |0,1> | |2,0> | |1,1> | |0,2> |
+-------+-------+-------+-------+-------+-------+
| |0,1> |  1/2  |  1/2  |   0   |   0   |   0   |
| |1,0> |  1/2  |  1/2  |   0   |   0   |   0   |
| |1,1> |   0   |   0   |  1/2  |   0   |  1/2  |
+-------+-------+-------+-------+-------+-------+

Congratulations, you have achieved this first tutorial! You can now continue with the documentation through
:ref:`Circuits`, :ref:`Computing Backends`, :ref:`States`, :ref:`Polarization` or you will go through more advanced
examples in the `Examples` section starting with :ref:`Getting started with Perceval`.
