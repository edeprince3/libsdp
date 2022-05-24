# libsdp
a library of semidefinite programming solvers

[Installation](#installation)

[Quickstart](#quickstart)

[The SDP Problem](#the-sdp-problem)

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
In this example, the SDP problem is expressed in the "SDPA" sparse format described here: http://euler.nmt.edu/~brian/sdplib/sdplib.pdf . We solve a problem that is identical to that given in the C/C++ interface example. You can see how that problem is represented in SDPA-style sparse format in 
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
where $x$ is the primal solution vector, $c$ is a vector that defines the objective function being minimized, and $A$ and $b$ represent the constraint matrix and vector, respectively, which encode the linear constraints applied in the problem. If you are familiar with the SDPA representation of the SDP, our formulation is equivalent (to within a sign) what SDPA considers to be the dual problem.

### Representing the SDP in Python

### Representing the SDP in C/C++

The C/C++ interface to libsdp operates through callback functions, defined by the user. To solve your SDP using this interface, you must provide the following information to the library:

  - a list of dimensions of each block of the primal solution vector
  - the vector b that defines the right-hand side of the constraints (Ax = b)
  - the vector c that defines the objective function
  - a callback function to evaluate the action of the constraint matrix on a vector (Au)
  - a callback function to evlauate the action of the transpose of the constraint matrix on a vector (A^Tu)
  - a callback function to monitor the progress of the solver

For additional detailes, see the sample code in libsdp/examples/c_interface/main.cc, which defines each of these quantities and passes them to the libsdp solver.
