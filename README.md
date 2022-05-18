# libsdp
a library of semidefinite programming solvers

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
The package will attemt to locate BLAS libraries and install libLBFGS. Assuming these steps are successful, you can then build the library
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

### Python interface 

An example using the Python interface to libsdp is provided in
```
libsdp/examples/python_interface
```
In this example, the SDP problem is expressed in the "SDPA" sparse format described here: http://euler.nmt.edu/~brian/sdplib/sdplib.pdf . That pdf also describes a library of test SDP problems, and SPDA-tyle input files for which can be found here: http://euler.nmt.edu/~brian/sdplib/. The example for the Python interface to libsdp corresponds to the "truss1" test case. You can run that test by first building the Python interface to the library
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
    filename = 'truss1.dat-s'
```
with the corresponding file name in libsdp/examples/python_interface/sdpa_format.py
