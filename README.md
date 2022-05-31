# libsdp
a library of semidefinite programming solvers

[Installation](#installation)

[Quickstart](#quickstart)

[The SDP Problem](#the-sdp-problem)

[Methods and Functionality](#methods-and-functionality)


## Installation
Installing libsdp requires cmake.  To install, first clone the package
```
git clone git@github.com:edeprince3/libsdp.git
```
Then, to build the library's C/C++ interface
```
cd libsdp
cmake .
```
The package will attempt to locate BLAS and install libLBFGS. Assuming these steps are successful, you can then build the library
```
make 
```
Alternatively, to build the library's Python interface
```
cd libsdp
cmake . -DBUILD_PYTHON_INTERFACE=true
make
```
Note that the Python interface also requires BLAS and libLBFGS.

## Quickstart

### C/C++ interface

An example using the C/C++ interface to libsdp is provided in
```
libsdp/examples/c_interface
```
This example is written as a project that downloads/builds/links to libsdp automatically. You should be able to execute this example by
```
cd libsdp/examples/c_interface
cmake .
make
./libsdp_c_interface rrsdp (or bpsdp)
```
where "rrsdp" and "bpsdp" refer to different SDP solvers. 

In order to use the C/C++ interface to libsdp for other problems, you must develop a few callback functions that define the problem (see below for more details).

### Python interface (SDPA formatted input files)

An example using the Python interface to libsdp is provided in
```
libsdp/examples/python_interface
```
In this example, the SDP problem is expressed in the "SDPA" sparse format described here: http://euler.nmt.edu/~brian/sdplib/sdplib.pdf . We solve a problem that is identical to that given in the C/C++ interface example. You can see how that problem is represented in SDPA-style format in 
```
libsdp/examples/python_interface/c_example.in
```
You can run the corresponding test by first building the Python interface to the library
```
cd libsdp
cmake . -DBUILD_PYTHON_INTERFACE=true
make
```
and then
```
cd examples/python_interface
python sdpa_format.py
```
Other SDP problems can be solved using the Python interface by developing a suitable SDPA-style input file and replacing
```
    filename = 'c_example.in'
```
with the corresponding file name in libsdp/examples/python_interface/sdpa_format.py

### Python interface (Psi4)

An example in which libsdp interfaces directly with the Psi4 electronic structure package through Python can be found in
```
libsdp/examples/psi4_interface/psi4_v2rdm.py
```
In this case, we are solving an electronic structure problem by the variational two-electron reduced density matrix (v2RDM) approach, and the Psi4 package provides the necessary one- and two-electron integrals. We use SDPA-style sparse format to define the problem, but the interface is direct through Python, rather than through an input file as in the previous example. To run this example, you need to install Psi4 (see https://psicode.org/).

### Python interface (PySCF)

An example in which libsdp interfaces directly with the PySCF electronic structure package through Python can be found in
```
libsdp/examples/pyscf_interface/pyscf_v2rdm.py
```
This example is the same as the Psi4 one, except the required one- and two-electron integrals are taken from the PySCF package. To run this example, you need to install PySCF (see https://pyscf.org/).

## The SDP problem

We express the primal form of the semidefinite programming (SDP) problem as
$$ {\rm min} ~ x^T c $$
$$ Ax ~ = ~ b$$
$$ x ~ \succeq ~ 0$$
where $x$ is the primal solution vector, $c$ is a vector that defines the objective function being minimized, and $A$ and $b$ represent the constraint matrix and vector, respectively, which encode the linear constraints applied in the problem. If you are familiar with the SDPA representation of the SDP, our formulation is equivalent (to within a sign) to what SDPA considers to be the dual problem.

### Representing the SDP problem in Python

You can use the Python interface to libsdp in one of two ways. First, you can use an SDPA-style input file, as is done in 
```
libsdp/examples/python_interface
```
Alternatively, you can interface with the library directly, as is done in the Psi4 and PySCF examples. To do so, you must provide the library

  - a list of dimensions of each block of the primal solution vector
  - the vector b that defines the right-hand side of the constraints (Ax = b)
  - the vector c that defines the objective function, in SDPA sparse format
  - each row of the constraint matrix A, in SDPA sparse format

Note that c and the rows of A are passed as a single list, and each item in this list is an "sdp_matrix" object, which is a struct referring to c or a row of A. This struct contains

  - a list of block numbers for the particular blocks of the primal solution to which the constraint refers (unit offset)
  - a list of the rows corresponding to the elements in the blocks of the primal solution to which the constraint refers (unit offset)
  - a list of the columns corresponding to the elements in the blocks of the primal solution to which the constraint refers (unit offset)
  - a list of values by which the elements referenced by the block/row/column combination should be scaled for this constraint

For additional detailes, see the sample code in 
```
libsdp/examples/psi4_interface/psi4_v2rdm.py
```
which defines each of these quantities and passes them to the libsdp solver.


### Representing the SDP problem in C/C++

The C/C++ interface to libsdp operates through callback functions, defined by the user. To solve your SDP using this interface, you must provide the following information to the library:

  - a list of dimensions of each block of the primal solution vector
  - the vector b that defines the right-hand side of the constraints (Ax = b)
  - the vector c that defines the objective function
  - a callback function to evaluate the action of the constraint matrix on a vector (Au)
  - a callback function to evlauate the action of the transpose of the constraint matrix on a vector (A^Tu)
  - a callback function to monitor the progress of the solver

For additional detailes, see the sample code in 
```
libsdp/examples/c_interface/main.cc
```
which defines each of these quantities and passes them to the libsdp solver.

## Methods and Functionality

The following methods and objects are exposed at the Python layer:

#### libsdp.sdp_matrix()

Returns a struct for SDPA-style sparse matrix entries. Each row of the sparse matrix has elements that map to the elements of the primal solution vector for evaluating matrix-vector products (e.g, for Ax = b). A row of A can be encoded with the following elements of the sdp_matrix() struct:

  - block_number: a list of blocks in which particular elements of the primal solution reside (unit offset)
  - row: a list of rows in the corresponding block where particular elements of the primal solution reside (unit offset)
  - column: a list of columns in the corresponding block where particular elements of the primal solution reside (unit offset)
  - value: a list of values by which the element corresponing to the block/row/column label should be scaled (the actual entries in the given row of A)

#### libsdp.sdp_options()

Returns an options object for controlling the SDP algorithm, convergence parameters, etc. Members include:

  - sdp_algorithm: the SDP algorithm to be used (valid options are SDPAlgorithm.BPSDP and SDPAlgorithm.RRSDP, see below)
  - sdp_error_convergence: convergence in ||Ax-b|| for RRSDP and convergence in ||Ax-b|| and ||A^Ty - c + z|| for BPSDP. Here, y and z are dual solution vectors optimized internally by libsdp.
  - sdp_objective_convergence: the convergence in x.c in RRSDP and the primal-dual gap |x.c - b.y| in BPSDP
  - cg_maxiter: maximum number of iterations for the conjugate gradient optimization of the dual solution vector (BPSDP only)
  - mu_update_frequency: number of macrocycles to perform before updating the penalty parameter (BPSDP only)
  - penalty_parameter_scaling: factory by which penalty parameter should be scaled in each macrocycle (RRSDP only)
  - maxiter: the maximum number of macrocycles

The options object also defines the enums used to specify sdp_algorithm:

  - SDPAlgorithm.BPSDP: an enum denoting the boundary-point SDP algorithm (BPSDP)
  - SDPAlgorithm.RRSDP: an enum denoting the matrix-factorization-based SDP algorithm (RRSDP)

Default settings can be found in libsdp/include/sdp_solver.h

#### libsdp.sdp_solver( libsdp.options() )

Returns a solver object for performing the actual SDP optimization. Takes an options object as input. A minimal workflow would be:

```
import libsdp

options = libsdp.sdp_options()

# change options if you don't like the defaults

sdp = libsdp.sdp_solver(options)

b = [] # ... fill with appropriate elements
F = [libsdp.sdp_matrix()] # ... fill with c, then rows of A
dimensions = [] # fill with block dimensions of primal solution
maxiter = 100000 # maximum number of iterations ... can be used to force BPSDP to return early

x = sdp.solve(b,F,dimensions,maxiter)
```

