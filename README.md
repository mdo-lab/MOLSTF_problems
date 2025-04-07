# README for Multi-objective L-shaped Test Functions

This code provides Python implementations of the multi-objective L-shaped test
functions described in the paper:

    Kenny, A., Ray, T., Singh, H. K., 2025. 
    "Multi-objective L-shaped Test Functions" 
    Published in the proceedings of GECCO '25.

See the paper for more details on problem formulation.

## Installation and Dependencies

The code is written in Python 3.11.9 and requires the following packages:

numpy
pymoo
matplotlib (for examples)

The code can be installed by unzipping it and running `pip install .` from the
root directory. This will install the package and its dependencies (run 
`python setup.py clean` to remove unnecessary build artifacts). The package can
then be imported using `import molstf_problems`.

## Implementation and Usage

The Python implementation of the multi-objective L-shaped test functions 
provided here is designed to interface with the Pymoo optimization toolbox. The 
files `dtlz2_alpha.py` and `rlsf.py` provide the class definitions for the two 
sets of problems.

Once imported, the problems can be instantiated and used as a standard Pymoo 
problem. For example, to create an instance of the DTLZ2α problem with 2 
objectives and 10 variables, use the following code:

`problem = DTLZDAlpha(n_var=10, n_obj=2)`

The problems can be instantiated with default parameters or with 
user-specified ones. 

The DTLZ2α class of problems has the following parameters:
- `n_var`: the number of decision variables
- `n_obj`: the number of objectives
- `alpha`: controls the sharpness of the Pareto front bend
- `beta`: controls the range of the feasible region
- `gamma`: controls the scaling of the entire function
- `x0`: the value of x to obtain the Pareto front
- `g_func`: the auxiliary function

The RLSF class of problems has the following parameters:
- `n_var`: the number of decision variables
- `n_obj`: the number of objectives
- `alpha`: controls the sharpness of the Pareto front bend
- `beta`: controls the range of the feasible region
- `gamma`: controls the scaling of the entire function
- `eta`: controls the shape of the feasible region
- `x0`: the value of x to obtain the Pareto front
- `g_func`: the auxiliary function

See `nsga2_example.ipynb` in the `examples` directory for an example of how 
to use the problems with Pymoo's NSGA-II algorithm.
