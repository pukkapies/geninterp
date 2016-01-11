__author__ = 'Kevin Webster'
# This module is written to help in the implementation of the Wendland function kernel

import copy
import math
import numpy as np
import geninterp.factors as factors_module

# def name_of_object(arg):
#     # check __name__ attribute (functions)
#     try:
#         return arg.__name__
#     except AttributeError:
#         pass
#
#     for name, value in globals().items():
#         if value is arg and not name.startswith('_'):
#             return name

#TODO: Change coefficients to numpy 'int64' type to help avoid scalar overflow errors

class Poly:
    """
    Polynomial class implements polynomials as dictionaries.
    Supplies methods for differentiation and integration
    """

    def __init__(self, degree):
        self.degree = degree
        self.dict = {}
        for index in range(degree + 1):
            self.dict[index] = 0

    def __str__(self):
        return'Poly' + str(self.dict)

    __repr__ = __str__

    def coeff(self, index, coeff):
        if index > self.degree:
            print("ERROR: Index too high for polynomial of degree ", self.degree)
            exit(1)
        self.dict[index] = coeff

    def differentiate(self):
        self.verify()
        self.dict[0] = 0
        if self.degree > 0:
            for index in range(self.degree):
                if index + 1 in self.dict:
                    self.dict[index] = self.dict[index + 1] * (index + 1)
                    del self.dict[index + 1]
            self.degree -= 1

    def integrate(self):
        for index in reversed(range(self.degree + 1)):
            self.dict[index + 1] = self.dict[index] / (index + 1)
        self.dict[0] = 0  # Coefficient of integration is zero
        self.degree += 1

    def verify(self):
        """
        Verifies the Poly object by checking for any zero coefficients
        and checking the degree attribute is correct.
        Represent the zero polynomial as a zero scalar (degree 0)
        """
        empty_indices = []
        for key in self.dict:
            if self.dict[key] == 0:
                empty_indices.append(key)
        for index in empty_indices:
            del self.dict[index]
        if not self.dict:  # If the dict is now empty
            self.dict[0] = 0
        if self.degree != max(self.dict.keys()):
            print("WARNING: Incorrect degree for Poly object ", self, ": resetting")
            self.degree = max(self.dict.keys())

    def __add__(self, other):
        """
        Adds two polynomials of class Poly
        :param other: Poly object
        :return: Poly object that is the sum of self and other
        """
        assert isinstance(other, Poly)
        self.verify()
        other.verify()
        sum_poly = Poly(max(self.degree, other.degree))
        for index in range(sum_poly.degree + 1):
            if index not in self.dict:
                self.dict[index] = 0
            if index not in other.dict:
                other.dict[index] = 0
            sum_poly.dict[index] = self.dict[index] + other.dict[index]
            if sum_poly.dict[index] == 0:
                del sum_poly.dict[index]
        sum_poly.degree = max(sum_poly.dict.keys())
        return sum_poly

    def __sub__(self, other):
        """
        Subtracts other from self
        :param other: Poly object
        :return: Poly object that is self - other
        """
        assert isinstance(other, Poly)
        self.verify()
        other.verify()
        diff_poly = Poly(max(self.degree, other.degree))
        for index in range(diff_poly.degree + 1):
            if index not in self.dict:
                self.dict[index] = 0
            if index not in other.dict:
                other.dict[index] = 0
            diff_poly.dict[index] = self.dict[index] - other.dict[index]
            if diff_poly.dict[index] == 0:
                del diff_poly.dict[index]
        diff_poly.degree = max(diff_poly.dict.keys())
        return diff_poly

    def __mul__(self, other):
        """
        Multiplies self by other
        :param other: Poly object
        :return: Poly object self * other
        """
        self.verify()
        if isinstance(other, Poly):
            other.verify()
            return_poly = Poly(0)
            polylist = []
            for index in range(self.degree + 1):
                if index in self.dict:
                    polymon_temp = Poly(index)
                    polymon_temp.coeff(index, self.dict[index])
                    polylist.append(poly_monomial_mult(polymon_temp, other))
            for i in range(len(polylist)):
                return_poly = return_poly + polylist[i]
            return return_poly
        else:       # Scalar multiplication
            if other == 0:
                return Poly(0)
            else:
                return_poly = Poly(0)
                return_poly.degree = self.degree
                for index in self.dict:
                    return_poly.dict[index] = other * self.dict[index]
                return return_poly

    __rmul__ = __mul__

"""
# Example
poly = Poly(2)
poly.coeff(0, 3)
poly.coeff(1, 1)
poly.coeff(2, 4)

print("Degree = ", poly.degree)
print(poly.dict)


print("integrating...")
poly.integrate()

print("Degree = ", poly.degree)
print(poly.dict)

print("differentiating...")
poly.differentiate()

print("Degree = ", poly.degree)
print(poly.dict)
"""


# def add_poly(poly1, poly2):     # Redundant function with operator overloading
#     """
#     Adds two polynomials of class Poly
#     :param poly1: Poly object
#     :param poly2: Poly object
#     :return: Poly object that is the sum of poly1 and poly2
#     """
#     assert isinstance(poly2, Poly)
#     assert isinstance(poly1, Poly)
#     poly1.verify()
#     poly2.verify()
#     sum_poly = Poly(max(poly1.degree, poly2.degree))
#     for index in range(sum_poly.degree + 1):
#         if index not in poly1.dict:
#             poly1.dict[index] = 0
#         if index not in poly2.dict:
#             poly2.dict[index] = 0
#         sum_poly.dict[index] = poly1.dict[index] + poly2.dict[index]
#         if sum_poly.dict[index] == 0:
#             del sum_poly.dict[index]
#     sum_poly.degree = max(sum_poly.dict.keys())
#     return sum_poly
#
#
# def subtract_poly(poly1, poly2):      # Redundant function with operator overloading
#     """
#     Subtracts poly2 from poly1
#     :param poly1: Poly object
#     :param poly2: Poly object
#     :return: Poly object that is poly1 - poly2
#     """
#     assert isinstance(poly2, Poly)
#     assert isinstance(poly1, Poly)
#     poly1.verify()
#     poly2.verify()
#     diff_poly = Poly(max(poly1.degree, poly2.degree))
#     for index in range(diff_poly.degree + 1):
#         if index not in poly1.dict:
#             poly1.dict[index] = 0
#         if index not in poly2.dict:
#             poly2.dict[index] = 0
#         diff_poly.dict[index] = poly1.dict[index] - poly2.dict[index]
#         if diff_poly.dict[index] == 0:
#             del diff_poly.dict[index]
#     diff_poly.degree = max(diff_poly.dict.keys())
#     return diff_poly
#
#
# def scalar_mult(poly, scalar):      # Redundant function with operator overloading
#     """
#     Multiplies poly by a scalar
#     :param poly: Poly object
#     :param scalar: a scalar
#     :return: Poly object that is scalar * poly
#     """
#     assert isinstance(poly, Poly)
#     poly.verify()
#     if scalar == 0:
#         return Poly(0)
#     else:
#         mult_poly = copy.deepcopy(poly)
#         for index in mult_poly.dict:
#             mult_poly.dict[index] = scalar * poly.dict[index]
#         return mult_poly


def poly_monomial_mult(poly_mon, poly):
    """
    Multiplies a monomial by poly
    :param poly_mon: Poly object with just one nonzero coefficient at poly_hom.degree
    :param poly: Poly object
    :return: Poly object that is poly_mon * poly
    """
    assert isinstance(poly_mon, Poly)
    assert isinstance(poly, Poly)
    poly_mon.verify()
    poly.verify()
    for key in poly_mon.dict:
        if key != poly_mon.degree:
            assert poly_mon.dict[key] == 0
        else:
            assert poly_mon.dict[key] != 0
    return_poly = poly * poly_mon.dict[poly_mon.degree]
    if poly_mon.degree > 0:
        for index in reversed(range(poly.degree + 1)):
            return_poly.dict[index + poly_mon.degree] = return_poly.dict[index]
            del return_poly.dict[index]
        return_poly.degree += poly_mon.degree
    return return_poly

#
# def poly_mult(poly1, poly2):          # Redundant function with operator overloading
#     """
#     Multiplies poly1 by poly2
#     :param poly1: Poly object
#     :param poly2: Poly object
#     :return: Poly object poly1 * poly2
#     """
#     assert isinstance(poly1, Poly)
#     assert isinstance(poly2, Poly)
#     poly1.verify()
#     poly2.verify()
#     return_poly = Poly(0)
#     polylist = []
#     for index in range(poly1.degree + 1):
#         if index in poly1.dict:
#             polymon_temp = Poly(index)
#             polymon_temp.coeff(index, poly1.dict[index])
#             polylist.append(poly_monomial_mult(polymon_temp, poly2))
#     for i in range(len(polylist)):
#         return_poly = add_poly(return_poly, polylist[i])
#     return return_poly


"""
# Example
poly1 = Poly(3)
poly1.coeff(0, 3)
poly1.coeff(1, 1)
poly1.coeff(2, 4)
poly1.coeff(3,2)

poly2 = Poly(3)
poly2.coeff(0, 1)
poly2.coeff(1, 10)
poly2.coeff(2, 6)
poly2.coeff(3,2)

poly3 = Poly(2)
poly3.coeff(0, 1.5)
poly3.coeff(1, 10)
poly3.coeff(2, 6)

polymon = Poly(4)
polymon.coeff(4,3)

print("poly1 = ", poly1)
print("poly3 = ", poly3)
print("polymon = ", polymon)
print("polymon * poly3 = ", poly_monomial_mult(polymon, poly3).dict)
print("polymon * poly3 = ", polymon * poly3)
print("poly1 * poly3 = ", poly1 * poly3)

print("poly1 = ", poly1.dict)
print("poly3 = ", poly3.dict)
print("poly1 + poly3 = ", poly3 + poly1)
print("poly1 - poly2 = ", poly1 - poly2)
print("2.5 * poly1 = ", poly1 * 2.5)
"""


def integrate_by_parts(poly, d, divisor):
    """
    Function to integrate poly(r) * (1-r)^d by parts
    Returns a 'definite' part
    and an 'indefinite' part to be integrated again
    :param poly: Poly object
    :param d: degree of the second (degenerate) polynomial in integrand
    :param divisor: list containing divisors of the integral (stored separately)
    :return: a list of 2 lists containing Poly, degenerate poly degree and divisor list
            for definite and indefinite parts respectively
            The total divisor is the product of the divisors in the divisor list.
            These are stored separately to keep integer coefficients.
    """
    assert d >= 1
    assert isinstance(divisor, list)
    poly.verify()
    if poly.degree == 0:
        if poly.dict[0] == 0:
            print("ERROR: Zero polynomial entered into integrate_by_parts")
            exit(1)
        else:
            divisor2 = d + 1
            divisor.append(divisor2)
            return [[poly, d+1, divisor], []]
    else:
        copypoly = copy.deepcopy(poly)
        copypoly.differentiate()
        divisor2 = d + 1
        divisor.append(divisor2)
        return [[poly, d+1, copy.deepcopy(divisor)], [copypoly, d+1, copy.deepcopy(divisor)]]

"""
# Example
polyex = Poly(2)
polyex.coeff(1, 1)
polyex.coeff(2, 5)
print('polyex = ', polyex)

integ = integrate_by_parts(polyex, 5, [])
print("integ = ", integ)
print("integrate_by_parts(polyex, 5, 1)[0][0].degree = ", integ[0][0].degree)
print("integrate_by_parts(polyex, 5, 1)[1][0].degree = ", integ[1][0].degree)

print('')
polyex = Poly(0)
polyex.coeff(0, 2)
print('polyex = ', polyex)
integ = integrate_by_parts(polyex, 5, [5,6])
print("integ = ", integ)
print("integrate_by_parts(polyex, 5, 30)[0][0].degree = ", integ[0][0].degree)
"""

def nonintersect_elements(list1, list2):
    """
    Returns a list of elements in list1 that are not in list2
    :param list1: a list
    :param list2: a list
    :return: a list of elements in list1 that are not in list2
    """
    temp1 = copy.deepcopy(list1)
    temp2 = copy.deepcopy(list2)
    for item in temp2:
        if item in temp1:
            temp1.remove(item)
    return temp1


def intersect_elements(list_of_lists):
    """
    Computes intersecting elements of sublists
    :param list_of_lists: a list of lists
    :return: a list containing all intersecting elements
    """
    assert isinstance(list_of_lists, list)
    assert len(list_of_lists) != 0
    for sublist in list_of_lists:
        if len(sublist) == 0:
            return []
    intersection = list_of_lists[0]
    for element in intersection:
        for sublist in list_of_lists:
            if element not in sublist:
                intersection.remove(element)
                break
    return intersection


def binomial_coeff(n, k):
    """
    Computes binomial coefficient (n, k)
    :param n: an integer
    :param k: an integer, less than n
    :return: binomial coefficient (n, k)
    """
    assert n >= k
    if n == k:
        return 1
    elif k == 1:
        return n
    else:
        a = math.factorial(n)
        b = math.factorial(k)
        c = math.factorial(n-k)
        return a // (b * c)


def polydeg_to_poly(d):
    """
    Converts a degenerate polynomial (1-r)^d into a Poly object
    by expanding out terms
    :param d: degree of degenerate polynomial
    :return: Poly object equal to (1-r)^d (expanded)
    """
    return_poly = Poly(d)
    for index in range(d+1):
        return_poly.coeff(index, np.power(-1, index) * binomial_coeff(d, index))
    return_poly.verify()
    return return_poly

def list_product(alist):
    """
    Multiplies all elements of a list
    :param list: list of numbers
    :return: product of list elements
    """
    assert isinstance(alist, list)
    p = 1
    for i in alist:
        p *= i
    return p


def factorise(*args):
    """
    Factorises expressions of the form
    poly1(r) * (1-r)^d1 + poly2(r) * (1-r)^d2
    :param args: a sequence of lists of the form
    [Poly object1, degree1, divisor1list], [Poly object2, degree2, divisor2list],...
    :return: List of 3 elements: [poly, d, divisorlist] such that
            poly1(r) * (1-r)^d1 + poly2(r) * (1-r)^d2 = poly(r) * (1-r)^d / product(divisorlist)
    """
    poly_inputs = []
    degree_inputs = []
    divisor_inputs = []
    for arg in args:
        assert isinstance(arg, list)
        assert isinstance(arg[0], Poly)
        assert isinstance(arg[2], list)
        arg[0].verify()
        poly_inputs.append(arg[0])
        degree_inputs.append(arg[1])
        divisor_inputs.append(factors_module.prime_factors_list(arg[2]))
    num_args = len(poly_inputs)
    factors = [[1]]*num_args  # To collect product factors of each polynomial

    # Find minimum (degenerate) polynomial degree
    min_degree = min(degree_inputs)

    # The coefficients get large quickly, so keep product factors separate.
    # First calculate lowest common denominator
    divisor_input_copy = copy.deepcopy(divisor_inputs)
    lcm_divisors = []
    for index in range(len(divisor_inputs)):
        while len(divisor_input_copy[index]) != 0:
            temp = divisor_input_copy[index][0]
            lcm_divisors.append(temp)
            for item2 in divisor_input_copy:
                if temp in item2:
                    item2.remove(temp)
    lcm_divisors = factors_module.prime_factors_list(lcm_divisors)

    # Collect polynomial inputs over a common divisor
    for i in range(num_args):
        #if len(nonintersect_elements(lcm_divisors, divisor_inputs[i])) != 0:
        factors[i] = factors_module.prime_factors_list(nonintersect_elements(lcm_divisors, divisor_inputs[i]))
        #poly_inputs[i] = scalar_mult(poly_inputs[i], np.prod(nonintersect_elements(lcm_divisors, divisor_inputs[i])))
        divisor_inputs[i] = lcm_divisors

    # Next multiply out individual polynomial inputs with (1-r)^(degree_inputs[i] - min_degree) and factor out any common factor
    # in the polynomial coefficients
    for i in range(num_args):
        poly_inputs[i] = poly_inputs[i] * polydeg_to_poly(degree_inputs[i] - min_degree)
        temp = factors_module.prime_factors(factors_module.gcdd(*[poly_inputs[i].dict[ind] for ind in poly_inputs[i].dict]))
        if len(temp) != 0:
            for index in poly_inputs[i].dict:
                for t in temp:
                    assert not poly_inputs[i].dict[index] % t
                    poly_inputs[i].dict[index] //= t
        factors[i].extend(temp)
        factors[i].sort()
        degree_inputs[i] = min_degree   # Bookkeeping, not used after this

    # For each factor in lcm_divisors, check if it is also in each factors[i] list
    HCF = []
    for item in lcm_divisors:
        success = 1  # Assume item is in all factors sublists
        for i in range(num_args):
            if item not in factors[i]:
                success = 0  # item is not in factors[i]
                break
        if success == 1:   # item is in all factors sublists
            HCF.append(item)
            for j in range(num_args):
                factors[j].remove(item)
    lcm_divisors = nonintersect_elements(lcm_divisors, HCF)

    """
    # There are likely to be more factors that cancel after adding polynomials together. Adding the polynomials
    # together first can produce large coefficients, so calculate remainders of each polynomial coefficient
    # to cancel additional factors before adding the polynomials
    more_factors = []
    # Calculate maximum degree of poly_inputs
    max_degree = 0
    for index in range(len(poly_inputs)):
        if max_degree < max(poly_inputs[index].dict.keys()):
            max_degree = max(poly_inputs[index].dict.keys())


    #print('lcm_divisors = ', lcm_divisors)
    #print('poly_inputs = ', poly_inputs)
    #print('factors = ', factors)

    lolol = []  # One sublist for each degree, each sublist is a list of list of factors/coeffs
    for deg in range(max_degree + 1):
        lolol.append([list(tuple([f for f in factors[i]]) + (poly_inputs[i].dict[deg],))
                      for i in range(len(factors)) if deg in poly_inputs[i].dict])
    #print('lolol = ', lolol)

    remove_from_lcm_divisors = []
    for l in lcm_divisors:
        # Initialise a list of lists to collect which degrees have a factor l
        degree_factors = []
        for newlist in range(max_degree + 1):
            degree_factors.append([])   # Sublists will be mutated - this avoids all sublists of degree_factors sharing the same reference
        #print('l = ', l)
        for deg in range(max_degree + 1):
            lol = lolol[deg]
            #print('lol = ', lol)
            if factors_module.prod_sum_divisible(lol, l):
                degree_factors[deg].append(l)
            #print('degree_factors = ', degree_factors)
        intersection = intersect_elements(degree_factors)
        #print('intersection = ', intersection)
        if len(intersection):
            for deg in range(max_degree + 1):
                #print('deg = ', deg)
                lolol[deg] = factors_module.prod_sum_divide(lolol[deg], intersection)
            remove_from_lcm_divisors.extend(intersection)
            #print('cancelled common factor before adding, now lolol = ', lolol)
    #print('finished going through lcm_divisors, lolol = ', lolol)
    #print('remove_from_lcm_divisors = ', remove_from_lcm_divisors)
    for i in remove_from_lcm_divisors:
        lcm_divisors.remove(i)

    # Now multiply polynomials by remaining factors and add together
    output = [Poly(0), min_degree, lcm_divisors]
    for deg in range(max_degree + 1):
        p = Poly(deg)
        p.coeff(deg, factors_module.prod_sum_eval(lolol[deg]))
        output[0] = output[0] + p
    #print('output = ', output)
    #print('')
    """

    output = [Poly(0), min_degree, lcm_divisors]
    for i in range(num_args):
        output[0] = output[0] + (poly_inputs[i] * list_product(factors[i]))
    #print('output[0] = ', output[0])

    # Cancel any common factor
    polysum_factor_list = factors_module.prime_factors(factors_module.gcdd(*[output[0].dict[index] for index in output[0].dict]))
    HCF_after_adding = []
    for item in lcm_divisors:
        if item in polysum_factor_list:
            HCF_after_adding.append(item)
            polysum_factor_list.remove(item)
    #if len(HCF_after_adding) != 0:
    #    print("HCF_after_adding = ", HCF_after_adding)
    lcm_divisors =  nonintersect_elements(lcm_divisors, HCF_after_adding)
    output[2] = lcm_divisors
    for index in output[0].dict:
        for factor in HCF_after_adding:
            assert not output[0].dict[index] % np.prod(factor)
            output[0].dict[index] //= factor

    #common_factor = factors_module.gcdd(np.prod(lcm_divisors), *[output[0].dict[index] for index in output[0].dict])
    # assert np.prod(output[2]) % common_factor == 0
    # temp = common_factor
    # for div_ind in range(len(output[2])):
    #     temp2 = factors_module.gcd(temp, output[2][div_ind])
    #     output[2][div_ind] //= temp2
    #     temp //= temp2
    # assert temp == 1
    # for index in output[0].dict:
    #     assert output[0].dict[index] % common_factor == 0
    #     output[0].dict[index] //= common_factor


    # Remove any 1's from the divisors list (should be unnecessary)
    items_to_remove = []
    for index in range(len(output[2])):
        if output[2][index] == 1:
            items_to_remove.append(index)
    if len(items_to_remove) != 0:
        print("WARNING: Divisors list still has 1's")
        for index in sorted(items_to_remove, reverse=True):
            del output[2][index]

    #print('output = ', output)

    return output

"""
# Example
poly1 = Poly(1)
poly1.coeff(1, 1)
del poly1.dict[0]
print("poly1 = ", poly1)

poly2 = Poly(0)
poly2.coeff(0, 1)
print("poly2 = ", poly2)

fac = factorise([poly1, 5, [5]], [poly2, 6, [5,6]])
print('factorise([poly1, 5, [5]], [poly2, 6, [5,6]]) = ', fac)


print('')
# Example 2
poly1 = Poly(2)
poly1.coeff(1, 1)
poly1.coeff(2, 5)
print("poly1 = ", poly1)

poly2 = Poly(1)
poly2.coeff(0, 1)
poly2.coeff(1, 10)
print("poly2 = ", poly2)

poly3 = Poly(0)
poly3.coeff(0, 10)
print("poly3 = ", poly3)

fac = factorise([poly1, 6, [6]], [poly2, 7, [7,6]], [poly3, 8, [6,7,8]])
print('factorise([poly1, 6, [6]], [poly2, 7, [7,6]], [poly3, 8, [6,7,8]]) = ', fac)
"""

def poly_triple_to_poly(list_triple):
    """
    Converts a list of the form [Poly, degree, div_list] to Poly(r)*(1-r)^degree / prod(div_list) to Poly
    :param list_triple: a list of the form [Poly, degree, div_list]
    :return: Poly
    """
    assert isinstance(list_triple, list)
    assert len(list_triple) == 3
    degen_poly = Poly(list_triple[1])
    list_triple[0].verify()
    for index in range(list_triple[1] + 1):
        degen_poly.coeff(index, ((-1)**index)*binomial_coeff(list_triple[1], index))
    final_poly = degen_poly * list_triple[0]
    divisor = list_product(list_triple[2])
    for index in range(final_poly.degree + 1):
        if index in final_poly.dict:
            final_poly.dict[index] = final_poly.dict[index] / divisor
    final_poly.verify()
    return final_poly


def poly_eval(poly, r):
    """
    Evaluates a poly at r
    :param poly: Poly object
    :param r: evaluation argument
    :return: float
    """
    poly.verify()
    eval_output = 0
    for index in range(poly.degree + 1):
        if index in poly.dict:
            eval_output += poly.dict[index] * (r**index)
    return eval_output