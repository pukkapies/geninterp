__author__ = 'Kevin Webster'

from .polynomial import *

"""
Module provides standard Kernel functions for analysis
Inputs to kernels (and subfunctions) should be numpy arrays
Broadcasting is supported, the last dimension should equal the phase space dimension
Phase space dimension should be set in self.dim
"""


class Kernel(object):
    """
    Superclass for kernel functions
    """

    def __init__(self, dimension=None):
        self.dim = dimension

    def get_dim(self):
        return self.dim

    def at_x(self, x):
        """
        Defines kernel centred at a point
        :param x: 2D np.array containing point about which to centre kernel
        :return: function k_x that takes a single argument
        """

        def k_x(y):
            return self.eval(x, y)

        return k_x

    def eval(self, x1, x2):
        """
        Dummy function that should be overridden in subclasses
        """
        print("ERROR: Kernel not defined!")
        exit(1)


class Polynomial(Kernel):
    #TODO: Implement Polynomial kernel
    pass


class RBF(Kernel):
    """
    Subclass of Kernel consisting of all RBF kernels. The orbital derivative using
    these kernels can be represented using helper psi1 and psi2 functions.
    """

    def __init__(self, dimension=None):
        super().__init__(dimension=dimension)

    def psi1(self, x1, x2):
        """
        Dummy function that should be overridden in subclasses
        """
        print('ERROR: psi1 function not defined')
        exit(1)

    def psi2(self, x1, x2):
        """
        Dummy function that should be overridden in subclasses
        """
        print('ERROR: psi2 function not defined')
        exit(1)


class Gaussian(RBF):
    """
    Gaussian kernel
    """

    def __init__(self, variance, dimension=None):
        self.var = float(variance)  # assign as float to remove 'classic division' compatibility problems
        super().__init__(dimension=dimension)

    def __repr__(self):
        if self.dim:
            return 'Gaussian(%d, dimension=%d)' % (self.var, self.dim)
        else:
            return 'Gaussian(%d)' % (self.var)

    def __str__(self):
        return self.__repr__()

    def eval(self, x1, x2):
        """
        Ordinary function evaluation
        :param x1, x2: 1D np.arrays each containing a point to input to Gaussian kernel function
        :return: Gaussian kernel evaluation
        """
        #assert x1.ndim == 1 and x2.ndim == 1
        assert x1.shape[-1] == x2.shape[-1]   # Same phase space dimension
        assert self.dim == x1.shape[-1]
        dim_axis = max(x1.ndim, x2.ndim) - 1
        return np.exp(-(np.linalg.norm((x1 - x2) ** 2, axis=dim_axis) / self.var))


    def psi1(self, x1, x2):
        """
        Helper function for RBF orbital derivative interpolation
        :param x1: input vector, 1D numpy array
        :param x2: input vector, 1D numpy array
        :return: (dG/dr) / r where G is Gaussian kernel
        """
        return -2 * self.eval(x1, x2) / self.var

    def psi2(self, x1, x2):
        """
        Helper function for RBF orbital derivative interpolation
        :param x1: input vector, 1D numpy array
        :param x2: input vector, 1D numpy array
        :return: (d(psi1)/dr) / r
        """
        return 4 * self.eval(x1, x2) / (self.var ** 2)


class Wendland(RBF):
    """
    Wendland function kernel
    """
    def __init__(self, l, k, dimension=None, c = 1, calculation='exact'):
        """
        Initialiser for Wendland kernel
        :param l: Internal parameter. For normal applications, set equal to
                    floor(d/2) + k + 1
        :param k: Internal parameter. With above setting for l, kernel has smoothness
                    C^{2k} and generates the Sobolev RKHS W^\tau_2, with
                    \tau = k + (d+1)/2
        :param dimension: Phase space dimension
        :param c: Positive scaling parameter. Set to 1 by default
        :param calculation: exact - use only integer arithmetic, representation is exact up until the final
                                    polynomial evaluation (but produces overflow errors from k=12)
                            exact-normalised - same as exact, but does not divide by lcm_divisors
                            float - use floating point arithmetic, end result subject to rounding errors
                            scaled - produces functions equal to Wendland kernel up to a constant.
                                    Coefficients are scaled to stay within a suitable range
        :return: None
        """
        super().__init__(dimension=dimension)
        assert c > 0 and k >= 0 and l >= 1
        assert (not k % 1) and (not l % 1)  # Both integers
        self.c = float(c)  # assign as float to remove 'classic division' compatibility problems
        self.k = k
        self.l = l
        #self.psilist = [None] * (self.k+1)  # To store list of [psi_{l,0},...,psi_{l,k}]
                     # Each psi is stored as [Poly, degree, divisor], such that psi = Poly * (1-r)^degree / divisor
        # NB Both psi1 and psi2 are stored as if c=1. The scaling is included only at the evaluation step
        self.psilist = None     # To store the polynomial in the form [Poly, degree, divisor_list]
        self.psi1list = None
        self.psi2list = None    # This is d/dr psi1 - i.e. psi2 * r
        self.poly = None        # Final polynomial representation - equal to self.psilist expanded out (without c scaling)
        self.psi1poly = None    # psi1 polynomial representation - equal to self.psi1list expanded out (without c scaling)
        self.psi2poly = None    # Equal to self.psi2list expanded out (without scaling), divided by r. This is O(1/r),
                                # so is saved as a list [poly, const], such that psi2 = poly + (const / r)
        self.eval = self.psilk()
        self.psi1 = self.psi1calc()
        self.psi2 = self.psi2calc()
        assert calculation == 'exact' or calculation == 'exact-normalised' \
               or calculation == 'float' or calculation == 'scaled'
        self.calculation = calculation
        #TODO: Add functionality for exact-normalised, float and scaled

    def __repr__(self):
        if self.dim:
            return "Wendland(%d, %d, dimension=%d, c=%f, calculation='%s')" % \
                   (self.l, self.k, self.dim, self.c, self.calculation)
        else:
            return "Wendland(%d, %d, c=%f, calculation='%s')" % \
                   (self.l, self.k, self.c, self.calculation)

    def __str__(self):
        return self.__repr__()

    def psilk(self):
        """
        Wendland function is defined recursively, starting from psi_{l,0}.
        This function is called in the initialiser to generate evaluation function
        :return: Evaluation function
        """
        const_poly = Poly(0)
        const_poly.coeff(0, 1)  # Equal to constant 1

        t = Poly(1)
        t.coeff(1, 1)   # Equal to polynomial t - to be used on each iteration in the recursion

        self.psilist = [const_poly, self.l, []]  # psi_{l,0} - equal to 'degenerate' polynomial with degree l: (1-r)^l
        for i in range(1, self.k + 1):
            current_psi_addends = []
            #integrand = [poly_mult(t, self.psilist[i-1][0]), self.psilist[i-1][1], copy.deepcopy(self.psilist[i-1][2])]
            integrand = [t * self.psilist[0], self.psilist[1], copy.deepcopy(self.psilist[2])]
            while True:
                temp = integrate_by_parts(integrand[0], integrand[1], integrand[2])
                current_psi_addends.append(temp[0])
                if temp[1] == []:  # Integral has been completed
                    break
                else:
                    integrand = temp[1]
            #self.psilist[i] = factorise(*current_psi_addends)
            self.psilist = factorise(*current_psi_addends)
        
        self.poly = poly_triple_to_poly(self.psilist)

        # Convert self.psilist into a workable function
        def psilist_eval(x1, x2):
            #assert x1.ndim == 1 and x2.ndim == 1
            assert x1.shape[-1] == x2.shape[-1]   # Same phase space dimension
            assert self.dim == x1.shape[-1]
            dim_axis = max(x1.ndim, x2.ndim) - 1
            cr = self.c * np.linalg.norm(x1 - x2, axis=dim_axis)
            mask = (cr < 1)
            output = np.zeros(cr.shape)
            output[mask] = poly_eval(self.poly, cr[mask])
            return output

        return psilist_eval

    def psi1calc(self):
        """
        Helper function for RBF orbital derivative interpolation
        :param x1: input vector, 1D numpy array
        :param x2: input vector, 1D numpy array
        :return: (dG/dr) / r where G is Gaussian kernel
        """
        psi1_addends = []
        psi1_addends.append([-self.psilist[1] * self.psilist[0], self.psilist[1] - 1, self.psilist[2]])
        temp_poly = copy.deepcopy(self.psilist[0])
        temp_poly.differentiate()
        psi1_addends.append([temp_poly, self.psilist[1], self.psilist[2]])
        d_dr_psi = factorise(*psi1_addends)
        # d_dr_psi should be divisible by r
        if 0 in d_dr_psi[0].dict:
            #TODO: change condition when calculation is float or scaled
            assert d_dr_psi[0].dict[0] == 0     # Constant should be zero

        # Now divide by r
        for i in range(d_dr_psi[0].degree):
            d_dr_psi[0].dict[i] = d_dr_psi[0].dict[i + 1]
        del d_dr_psi[0].dict[d_dr_psi[0].degree]
        d_dr_psi[0].degree -= 1
        self.psi1list = d_dr_psi

        self.psi1poly = poly_triple_to_poly(self.psi1list)

        psi1poly_scaled = copy.deepcopy(self.psi1poly)
        for ind in psi1poly_scaled.dict:
            psi1poly_scaled.dict[ind] *= self.c**(ind + 2)


        def psi1eval(x1, x2):
            assert self.k >= 1  # Otherwise psi1 and psi2 are not defined
            #assert x1.ndim == 1 and x2.ndim == 1
            assert x1.shape[-1] == x2.shape[-1]   # Same phase space dimension
            assert self.dim == x1.shape[-1]
            dim_axis = max(x1.ndim, x2.ndim) - 1
            r = np.linalg.norm(x1 - x2, axis=dim_axis)
            mask = (self.c * r < 1)
            output = np.zeros(r.shape)
            output[mask] = poly_eval(psi1poly_scaled, r[mask])
            return output

        return psi1eval

    def psi2calc(self):
        """
        Helper function for RBF orbital derivative interpolation
        Returns function with parameters x1 and x2
        :param x1: input vector, 1D numpy array
        :param x2: input vector, 1D numpy array
        :return: (d(psi1)/dr) / r
        """
        psi2_addends = []
        psi2_addends.append([-self.psi1list[1] * self.psi1list[0], self.psi1list[1] - 1, self.psi1list[2]])
        temp_poly = copy.deepcopy(self.psi1list[0])
        temp_poly.differentiate()
        if not (temp_poly.degree == 0 and temp_poly.dict[0] == 0): # Zero polynomial
            psi2_addends.append([temp_poly, self.psi1list[1], self.psi1list[2]])
        d_dr_psi1 = factorise(*psi2_addends)
        self.psi2list = d_dr_psi1

        divisor = list_product(d_dr_psi1[2])
        d_dr_psi1 = poly_triple_to_poly([d_dr_psi1[0],d_dr_psi1[1],[]])

        # Now divide nonconstant terms by r - save constant in a separate variable
        if 0 in d_dr_psi1.dict:
            constant_coeff = d_dr_psi1.dict[0]
        else:
            constant_coeff = 0
        for i in range(d_dr_psi1.degree):
            if i+1 in d_dr_psi1.dict:
                d_dr_psi1.dict[i] = d_dr_psi1.dict[i+1]
                del d_dr_psi1.dict[i+1]
        d_dr_psi1.degree -= 1
        self.psi2poly = [(1/divisor) * d_dr_psi1, (1/divisor) * constant_coeff]

        psi2poly_scaled = copy.deepcopy(self.psi2poly)
        for ind in psi2poly_scaled[0].dict:
            psi2poly_scaled[0].dict[ind] *= self.c**(ind + 4)
        psi2poly_scaled[1] *= (self.c)**3

        def psi2eval(x1, x2):
            assert self.k >= 1  # Otherwise psi1 and psi2 are not defined
            #assert x1.ndim == 1 and x2.ndim == 1
            assert x1.shape[-1] == x2.shape[-1]   # Same phase space dimension
            assert self.dim == x1.shape[-1]
            dim_axis = max(x1.ndim, x2.ndim) - 1
            r = np.linalg.norm(x1 - x2, axis=dim_axis)
            mask = (self.c * r < 1) * (r != 0)
            output = np.zeros(r.shape)
            output[mask] = poly_eval(psi2poly_scaled[0], r[mask]) + (psi2poly_scaled[1] / r[mask])
            return output

        return psi2eval

    def differentiate(self, n):
        """
        Returns a new Wendland kernel object that is the nth derivative of the current Wendland kernel
        """
        return Wendland_deriv(self, n)


class Wendland_deriv(RBF):
    """
    Wendland function kernel derivative
    """
    def __init__(self, Wendlandkernel, derivs):
        """
        Initialiser for Wendland kernel
        :param l: Internal parameter. For normal applications, set equal to
                    floor(d/2) + k + 1
        :param k: Internal parameter. With above setting for l, kernel has smoothness
                    C^{2k} and generates the Sobolev RKHS W^\tau_2, with
                    \tau = k + (d+1)/2
        :param dimension: Phase space dimension
        :param c: Positive scaling parameter. Set to 1 by default
        :param calculation: exact - use only integer arithmetic, representation is exact up until the final
                                    polynomial evaluation (but produces overflow errors from k=12)
                            exact-normalised - same as exact, but does not divide by lcm_divisors
                            float - use floating point arithmetic, end result subject to rounding errors
                            scaled - produces functions equal to Wendland kernel up to a constant.
                                    Coefficients are scaled to stay within a suitable range
        :param derivs: number of derivatives to take
        :return: None
        """
        super().__init__(dimension=Wendlandkernel.dim)
        self.c = Wendlandkernel.c  # assign as float to remove 'classic division' compatibility problems
        self.k = Wendlandkernel.k
        self.l = Wendlandkernel.l
        self.derivs = derivs
        self.psilist = Wendlandkernel.psilist     # To store the polynomial in the form [Poly, degree, divisor_list]

        self.poly = None        # Final polynomial representation - equal to self.psilist expanded out (without c scaling)

        self.eval = self.psilk()

        self.calculation = Wendlandkernel.calculation
        #TODO: Add functionality for exact-normalised, float and scaled

    def __repr__(self):
        if self.dim:
            return "Wendland_deriv(%d, %d, dimension=%d, c=%f, calculation='%s', derivs=%d)" % \
                   (self.l, self.k, self.dim, self.c, self.calculation, self.derivs)
        else:
            return "Wendland_deriv(%d, %d, c=%f, calculation='%s', derivs=%d)" % \
                   (self.l, self.k, self.c, self.calculation, self.derivs)

    def __str__(self):
        return self.__repr__()

    def psilk(self):
        """
        Calculates representation in the form [Poly, degree, divisor_list]
        :return: Evaluation function
        """
        for counter in range(self.derivs):
            if self.psilist == 0:   # zero polynomial
                break
            polytemp = copy.deepcopy(self.psilist[0])
            polytemp.differentiate()
            if self.psilist[1] >= 1:    # Degenerate polynomial part has degree at least 1
                self.psilist = factorise([polytemp, self.psilist[1], self.psilist[2]],
                                         [-self.psilist[1] * self.psilist[0], self.psilist[1] - 1, self.psilist[2]])
            else:
                if polytemp.degree == 0 and polytemp.dict[0]==0:
                    self.psilist = 0    # zero polynomial
                else:
                    self.psilist = factorise([polytemp, self.psilist[1], self.psilist[2]])

        self.psilist[0] = (self.c**self.derivs) * self.psilist[0]

        if self.psilist != 0:
            self.poly = poly_triple_to_poly(self.psilist)
        else:
            self.poly = Poly(0)

        # Convert self.psilist into a workable function
        def psilist_eval(x1, x2):
            if self.dim != 1:
                print("Wendland_deriv: Evaluation function for dimensions higher than one is not yet implemented!")
                exit(1)
            #TODO: Implement this - for n derivatives and d dimensions the output should be a (n+1)-dimensional
            #TODO: array, where all but the first dimension have length d. First dimension is equal to number of input
            #TODO: points. Assume that the derivative is being taken with respect to the second input argument
            assert x1.shape[-1] == x2.shape[-1]   # Same phase space dimension
            assert self.dim == x1.shape[-1]
            dim_axis = max(x1.ndim, x2.ndim) - 1
            r = np.linalg.norm(x1 - x2, axis=dim_axis)
            cr = self.c * r
            mask = (cr < 1)
            output = np.zeros(cr.shape)
            output[mask] = poly_eval(self.poly, cr[mask])

            dr = np.sign(x2-x1)
            dr[dr==0] = 1   # These elements will be multiplied by zero for odd derivatives anyway
            dr = dr**self.derivs
            """
            r_inv = np.zeros(r.shape)
            mr = np.ma.array(r, mask=(r==0))
            r_inv[r!=0] = 1./(mr[~mr.mask])
            r_inv = np.expand_dims(r_inv, r_inv.ndim)   # reciprocal of r, leaving zero values at zero

            output = np.expand_dims(output, output.ndim)
            output = np.repeat(output, self.dim, axis=output.ndim - 1)
            output = output * r_inv * (x1 - x2)
            """
            output = np.expand_dims(output, output.ndim)
            output = output * dr

            assert output.shape[-1]==1  # Only for phase space dimension = 1, this will change for other dimensions
            output = output.squeeze(axis=(output.ndim - 1))

            return output

        return psilist_eval

"""
K31 = Wendland(3,1,c=0.5)
print('(l,k): ', (K31.l,K31.k), ' = ', K31.psilist)
print(K31.eval(np.array([3]),np.array([5.1])))
print(K31.psi1(np.array([4]),np.array([1.8])))
print(K31.psi2(np.array([-3]),np.array([-4.3])))
print("\n*****************************\n")
K42 = Wendland(4,2,c=0.5)
print('(l,k): ', (K42.l,K42.k), ' = ', K42.psilist)
print(K42.eval(np.array([3]),np.array([4.1])))
print(K42.psi1(np.array([4]),np.array([5.998])))
print(K42.psi2(np.array([-3]),np.array([-4])))
print("\n*****************************\n")
K53 = Wendland(5,3,c=0.5)
print('(l,k): ', (K53.l,K53.k), ' = ', K53.psilist)
print(K53.eval(np.array([3]),np.array([4.1])))
print(K53.psi1(np.array([4]),np.array([2.8])))
print(K53.psi2(np.array([-3]),np.array([-4.3])))
print("\n*****************************\n")
K64 = Wendland(6,4)
print('(l,k): ', (K64.l,K64.k), ' = ', K64.psilist)
print(list_product([2, 2, 2, 2, 2, 2, 2, 3, 3, 5, 7, 11, 13]))
"""