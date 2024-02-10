"""
Main package for libsdp

_libsdp is the name of the pybind module.

If it was also named libsdp we would have a namespace collision.

When importing libsdp it must go grab all the modules from libsdp
"""
from ._libsdp import *
