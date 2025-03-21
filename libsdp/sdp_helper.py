import libsdp
import numpy as np
import os

# minimal matrix object for sparse SDPA format
class sdp_matrix:
    def __init__(self,):
        self.row: list = []
        self.column: list = []
        self.block_number: list = []
        self.value: list = []
        self.id: list = []

def python_to_cpp_sdp_matrix(Fi: sdp_matrix):
    """
        convert a Python SDPA matrix object to a C++ one

        :param Fi: a Python SDPA matrix object (sdp_matrix)
        :return tf: a C++ SDPA matrix object (libsdp.sdp_matrix)
    """

    tF = libsdp.sdp_matrix()
    tF.row = Fi.row 
    tF.column = Fi.column 
    tF.value = Fi.value 
    tF.block_number = Fi.block_number 

    return tF

class sdp_solver:
    def __init__(self, options, F, dimensions):
        """
            a Python wrapper to the libsdp.sdp_solver object

            :param options: libsdp.options object
            :param F: a list of python sdp_matrix objects
            :param dimensions: a list of primal block dimensions for the problem
        """

        # convert python SDPA matrix objects to C++ ones
        self.F = []
        for Fi in F:
            self.F.append(python_to_cpp_sdp_matrix(Fi))
   
        self.options = options
        self.dimensions = dimensions

        self.sdp_solver = libsdp.sdp_solver(self.options, self.F, self.dimensions)

    def solve(self, b, maxdim, primal_rank = None) :

        self.maxdim = maxdim

        if primal_rank is None : 
            return self.sdp_solver.solve(b, maxdim)
        return self.sdp_solver.solve(b, maxdim, primal_rank)

    def get_ATu(self, u):
        return self.sdp_solver.get_ATu(u)

    def get_Au(self, u):
        return self.sdp_solver.get_Au(u)

    def get_y(self):
        return self.sdp_solver.get_y()

    def get_z(self):
        return self.sdp_solver.get_z()

    def get_c(self):
        return self.sdp_solver.get_c()

