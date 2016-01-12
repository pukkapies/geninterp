# geninterp
Python module for generalised interpolation

This module provides an implementation for generalised interpolation. This is a technique that allows to find functions that interpolate a given set of data points, and/or where values of some linear functionals are also given to be interpolated.

For example, a classical interpolation problem is where we have data points (x_i, y_i), i=1,...,m and we search for a function f such that f(x_i) = y_i for all i. In generalised interpolation, we define linear functionals; that is, a linear operator that takes a function as input and returns a real value. An example might be the linear functional h_i that is the derivative of a function at a given point x_i:

h_i(f) = df/dx (x_i).

Then an example of generalised interpolation is as follows: suppose we have data points (x_i, y_i), i=1,..,n+m and we want to find a function that satisfies f(x_i) = y_i, i=1,...,m and h_i(f) = df/dx (x_i) = y_i, i=m+1,...,m+n.

More generally, any linear functional can be applied in the process. Note that the standard function value interpolation corresponds to using the point evaluation linear functional.

Convergence estimates for the algorithm implemented here using Wendland kernels can be found in [1].

##Usage

The core module is generalised_interpolation.py. The process uses kernels to generate the function approximation spaces, the implementation for these are in kernels.py. This module contains the Wendland kernel - a radial basis function with compact support that generates Sobolev spaces as the RKHS, see [2]. These kernels are recursively defined, and an explicit derivation is implemented in this module, which in turn uses the modules polynomial.py and factors.py.

Examples of usage are available in geninterpolant_examples.py.

##Dependencies
* python 3
* numpy (1.6.0+)
* scipy (0.9.0+)

##References

1. P. Giesl and H. Wendland, 'Meshless collocation: error estimates with application to dynamical systems, SIAM J. Num. Anal. 45 No. 4 (2007), 1723-1741.
2. H. Wendland, Scattered Data Approximation, Cambridge Monogr. Appl.Comput. Math., Cambridge University Press, Cambridge, UK, 2005.
