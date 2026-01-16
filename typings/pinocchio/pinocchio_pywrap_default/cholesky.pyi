from __future__ import annotations
__all__: list[str] = ['computeMinv', 'decompose', 'solve']
def computeMinv(*args, **kwargs):
    """
    
    computeMinv( (pinocchio.pinocchio_pywrap_default.Model)Model, (pinocchio.pinocchio_pywrap_default.Data)Data) -> numpy.ndarray :
        Returns the inverse of the joint space inertia matrix using the results of the Cholesky decomposition
        performed by cholesky.decompose. The result is stored in data.Minv.
    """
def decompose(*args, **kwargs):
    """
    
    decompose( (pinocchio.pinocchio_pywrap_default.Model)Model, (pinocchio.pinocchio_pywrap_default.Data)Data) -> numpy.ndarray :
        Computes the Cholesky decomposition of the joint space inertia matrix M contained in data.
        The upper triangular part of data.M should have been filled first by calling crba, or any related algorithms.
    """
def solve(*args, **kwargs):
    """
    
    solve( (pinocchio.pinocchio_pywrap_default.Model)Model, (pinocchio.pinocchio_pywrap_default.Data)Data, (numpy.ndarray)v) -> numpy.ndarray :
        Returns the solution 
    $x
    $ of 
    $ M x = y 
    $ using the Cholesky decomposition stored in data given the entry 
    $ y 
    $.
    """
