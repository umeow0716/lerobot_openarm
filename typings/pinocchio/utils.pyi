from __future__ import annotations
import numpy as np
from numpy import linalg as npl
from pinocchio import pinocchio_pywrap_default as pin
import sys as sys
__all__: list = ['eye', 'fromListToVectorOfString', 'isapprox', 'matrixToRpy', 'mprint', 'np', 'npToTTuple', 'npToTuple', 'npl', 'rand', 'rotate', 'rpyToMatrix', 'zero']
def eye(n):
    ...
def fromListToVectorOfString(items):
    ...
def isapprox(a, b, epsilon = 1e-06):
    ...
def mprint(M, name = 'ans', eps = 1e-15):
    """
    
        Matlab-style pretty matrix print.
        
    """
def npToTTuple(M):
    ...
def npToTuple(M):
    ...
def rand(n):
    ...
def zero(n):
    ...
