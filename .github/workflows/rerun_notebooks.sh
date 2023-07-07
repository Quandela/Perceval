
my_function () {
    echo "$notebook"
    jupyter nbconvert --clear-output --inplace "$notebook"
    jupyter nbconvert --execute --to notebook "$notebook"
}

cd docs/source/notebooks/

notebook='2-mode Grover algorithm.ipynb'
my_function
notebook='Non-unitary components.ipynb' 
my_function
notebook='Shor Implementation.ipynb' 
my_function
notebook='QUBO.ipynb' 
my_function
notebook='Tutorial.ipynb' 
my_function
notebook='Boson Sampling with MPS.ipynb' 
my_function
notebook='Qiskit conversion.ipynb' 
my_function
notebook='Variational Quantum Eigensolver.ipynb' 
my_function
notebook='Reinforcement learning.ipynb' 
my_function
notebook='Differential equation solving.ipynb'
my_function
notebook='walkthrough-cnot.ipynb' 
my_function
notebook='Graph States and representation.ipynb' 
my_function
notebook='Rewriting rules in Perceval.ipynb'
my_function
