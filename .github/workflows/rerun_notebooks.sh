
my_function () {
    jupyter nbconvert --clear-output --inplace "$notebook.ipynb"
    jupyter nbconvert --execute --to notebook "$notebook.ipynb"
    rm "$notebook.ipynb"
    mv "$notebook.nbconvert.ipynb" "$notebook.ipynb"
}

cd docs/source/notebooks/

notebook='2-mode Grover algorithm'
my_function
notebook='Non-unitary components' 
my_function
notebook='Shor Implementation' 
my_function
notebook='QUBO' 
my_function
notebook='Tutorial' 
my_function
notebook='Boson Sampling with MPS' 
my_function
notebook='Qiskit conversion' 
my_function
notebook='Variational Quantum Eigensolver' 
my_function
notebook='Reinforcement learning' 
my_function
notebook='Differential equation solving'
my_function
notebook='walkthrough-cnot' 
my_function
notebook='Graph States and representation' 
my_function
notebook='Rewriting rules in Perceval'
my_function
