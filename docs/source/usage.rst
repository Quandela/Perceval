Usage
=====

Installation
------------

To use perceval, you can directly install the pip wheel in a virtual environment:

.. code-block:: bash

   (.venv) $ pip install perceval-quandela

Alternatively, if you are interested in contributing to the project - you can clone the project from github:

.. code-block:: bash

   (.venv) $ git clone https://github.com/quandela/Perceval
   (.venv) $ cd Perceval
   (.venv) $ python setup.py install # or develop for developers

At this point you can directly use Perceval. For following the tutorial below, we do recommend running it in a
jupyter notebook, or in the Python console of your favorite IDE (Spyder, Pycharm) but you can also run it in a
terminal python console.


First Circuit
-------------

Import the library and the components from `symb` library:

>>> import perceval as pcvl
>>> import perceval.lib.symb as symb

Defines a circuit as a simple beam-splitter, it is a 2-mode circuit

>>> c = symb.BS()
>>> c.m
2
>>> pcvl.pdisplay(c)

.. tip::

    The outcome of this last command will depend on your environment - see :ref:`Visualization`.

    .. list-table::
       :header-rows: 1
       :width: 100%

       * - Text Console
         - Jupyter Notebook
         - Within IDE (Pycharm, Spyder)
       * - .. image:: _static/img/terminal-screenshot.png
         - .. image:: _static/img/jupyter-notebook.png
         - .. image:: _static/img/ide-screenshot.png

Check the definition of the circuit, and the values of these parameters:

>>> pcvl.pdisplay(c.definition())
⎡cos(theta)               I*exp(-I*phi)*sin(theta)⎤
⎣I*exp(I*phi)*sin(theta)  cos(theta)              ⎦
>>> c.get_parameters(all_params=True)
[Parameter(name='phi', value=0, min=0, max=2*pi), Parameter(name='theta', value=pi/4, min=0, max=2*pi)]

Display the unitary matrix for these fixed parameters, and check it is unitary:

>>> pcvl.pdisplay(c.U)
⎡sqrt(2)/2    sqrt(2)*I/2⎤
⎣sqrt(2)*I/2  sqrt(2)/2  ⎦
>>> c.U.is_unitary()
True

Let us decide to send one photon on the lower left branch of the beam splitter. It is corresponding to the following
input state:

>>> input_state = pcvl.BasicState("|0,1>")
>>> input_state.n
1

Define a simulator for the circuit:

>>> backend = pcvl.BackendFactory().get_backend()
>>> simulator = backend(c)

Get the output state of the circuit for this input_state:

>>> print(simulator.evolve(input_state))
sqrt(2)*I/2*|1,0>+sqrt(2)/2*|0,1>

Sample some output states:

>>> for _ in range(10):
...    print(simulator.sample(pcvl.BasicState("|0,1>")))
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

Get the actual probability associated to each output state:

>>> pcvl.pdisplay(simulator.prob(input_state, pcvl.BasicState("|0,1>")))
1/2
>>> pcvl.pdisplay(simulator.prob(input_state, pcvl.BasicState("|1,0>")))
1/2
>>> pcvl.pdisplay(simulator.prob(input_state, pcvl.BasicState("|1,1>")))
0

Get the full probability distribution:

>>> ca = pcvl.CircuitAnalyser(simulator,
...                           [pcvl.BasicState([0, 1]), pcvl.BasicState([1, 0]), pcvl.BasicState([1, 1])], # the input states
...                           "*" # all possible output states that can be generated with 1 or 2 photons
...                          )
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
examples in the Examples section starting with :ref:`Perceval Detailed Walkthrough`.
