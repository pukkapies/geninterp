__author__ = 'Kevin Webster'

import copy

def prime_factors(n):
    """
    Factors n into primes
    :param n: An integer
    :return: A list of prime factors of n. If n = 1, return []
    """
    assert n >= 1
    if n == 1:
        return []
    else:
        i = 2
        factors = []
        while i * i <= n:
            if n % i:
                i += 1
            else:
                n //= i
                factors.append(i)
        if n > 1:
            factors.append(n)
        return sorted(factors)


def prime_factors_list(alist, sort=True):
    """
    Factors each element of alist into primes
    :param alist: a list of positive integers
    :param sort: option to sort the return list
    :return: a list containing all prime factors of all elements in alist
    """
    assert isinstance(alist, list)
    if len(alist) == 0:
        return []
    else:
        return_list = []
        for item in alist:
            return_list.extend(prime_factors(item))
        if sort:
            return sorted(return_list)
        else:
            return return_list


def gcd(a, b):
    """
    Return greatest common divisor using Euclid's Algorithm.
    :param a: a nonzero integer
    :param b: a nonzero integer
    """
    assert a != 0 and b != 0
    a_temp = abs(a)
    b_temp = abs(b)
    while b_temp:
        a_temp, b_temp = b_temp, a_temp % b_temp
    return a_temp


def gcdd(*args):
    """
    Return gcd of args
    :param args: A number of integers
    :return: gcd of args. If args is only 1 element, return that element
    """
    templist = []
    for arg in args:
        templist.append(arg)
    assert len(templist) >= 1
    if len(templist) == 1:
        return abs(templist[0])
    else:
        answer = gcd(templist[0], templist[1])
        for index in range(2, len(templist)):
            answer = gcd(answer, templist[index])
        return answer

def lcm(a, b):
    """
    Return lowest common multiple.
    :param a: an integer
    :param b: an integer
    """
    return a * b // gcd(a, b)


def lcmm(*args):
    """Return lcm of args."""
    answer = 1
    for arg in args:
        answer = lcm(answer, arg)
    return answer

"""
print('gcdd(21,6,12) = ', gcdd(21,6,12))
print('gcdd(*[40, 24, 36, 112]) = ', gcdd(*[40, 24, 36, 112]))

alist = [2, 330, 1420012, 42, 80]
print('alist = ', alist)
print('all prime factors (sorted) = ', prime_factors_list(alist))

print('')
alist[2:3] = prime_factors(alist[2])
print('3rd element factored: ',alist)
"""


def prod_div_quotient_rem(alist, divisor):
    """
    Calculates the quotient and remainder from dividing the product of elements in alist with divisor
    The method used does not actually multiply out the elements of alist - the function is intended
    to be used for situations where the product is very large
    :param alist: a list of nonnegative integers
    :param divisor: an integer
    :return: a list of two elements, the quotient and remainder.
    """
    assert isinstance(alist, list)
    for element in alist:
        assert element >= 0
    if len(alist) == 0:
        return [0, 0]
    elif len(alist) == 1:
        return [alist[0] // divisor, alist[0] % divisor]
    else:
        answer = [0, 0]
        quotient1 = alist[0] // divisor
        remainder1 = alist[0] % divisor
        for i in range(1, len(alist)):
            quotient2 = alist[i] // divisor
            remainder2 = alist[i] % divisor
            answer[0] = (quotient1 * quotient2 * divisor) + (quotient1 * remainder2) + (
                quotient2 * remainder1) + ((remainder1 * remainder2) // divisor)
            answer[1] = (remainder1 * remainder2) % divisor
            answer[0] += answer[1] // divisor
            quotient1 = answer[0]
            remainder1 = answer[1]
    return answer


def prod_div_rem(alist, divisor):
    """
    Calculates the remainder from dividing the product of elements in alist with divisor
    The method used does not actually multiply out the elements of alist - the function is intended
    to be used for situations where the product is very large
    :param alist: a list of nonnegative integers
    :param divisor: an integer
    :return: the remainder
    """
    assert isinstance(alist, list)
    for element in alist:
        assert element >= 0
    if len(alist) == 0:
        return 0
    elif len(alist) == 1:
        return alist[0] % divisor
    else:
        answer = 0
        remainder1 = alist[0] % divisor
        for i in range(1, len(alist)):
            remainder2 = alist[i] % divisor
            answer = (remainder1 * remainder2) % divisor
            remainder1 = answer
    return answer

"""
# Example
alist = [7,8,5]
print(prod_div_quotient_rem(alist, 3))      # [93, 1]
print(prod_div_rem(alist, 3))               # 1
alist2 = [2,3,7,8,9,8,9,10,11,13,17,23]
print(prod_div_quotient_rem(alist2, 29))    # [4197870918, 18]
print(prod_div_rem(alist2, 29))             # 18
"""


def prod_sum_divisible(list_of_lists, divisor):
    """
    Calculates if sum of products in list_of_lists is divisible by divisor, without
    multiplying out product terms (intended for large numbers)
    :param list_of_lists: Each sublist represents a product
    :param divisor: an integer
    :return: Boolean
    """
    assert isinstance(list_of_lists, list)
    lol_copy = copy.deepcopy(list_of_lists)
    remainder = 0
    for l in lol_copy:
        # First check if all elements are nonnegative
        sign = 1
        for index in range(len(l)):
            if l[index] < 0:
                sign *= -1
                l[index] *= -1
        if sign == 1:
            remainder += prod_div_rem(l, divisor)
        else:
            remainder -= prod_div_rem(l, divisor)
        remainder = remainder % divisor
    return remainder == 0

"""
# Example
lol = [[7,8,5], [8, 10, -1], [10]]
print(prod_sum_divisible(lol, 6))   # True
print(prod_sum_divisible(lol, 8))   # False
lol2 = [[9,10,11,16], [10,11,-48], [11, 96, 1], [96, -1, 1]]
print(prod_sum_divisible(lol2, 90)) # True
print(prod_sum_divisible(lol2, 7))  # False
lol3 = [[15,14,13,12,11,384], [15,14,13,12,-1920], [15,14,13,7680], [15,14,-23040], [15,46080], [-46080], []]
print(prod_sum_divisible(lol3, 15*14*13*12*11)) # True
print(prod_sum_divisible(lol3, 49)) # False
"""

def prod_sum_divide(list_of_lists, divisorlist):
    """
    Divides sum of products in list_of_lists by divisor, without multiplying out products
    !!!Assumes that sum of products in list_of_lists is divisible by prod(divisor)!!!
    :param list_of_lists: a list of lists of integers
    :param divisorlist: a list of positive divisors to divide into list_of_lists
    :return: a list of lists that is equal to sum([prod(sublist) for sublist in list_of_lists]) / prod(divisor)
    """
    assert isinstance(list_of_lists, list)
    lol_copy = []
    # First convert all sublists into prime factors. Any products that are negative get -1 at the beginning
    # Any empty sublists are removed
    for sublist in list_of_lists:
        sign = 1
        zero_flag = 0 # To catch any zeros in sublists
        new_sublist = []
        for index in range(len(sublist)):
            if sublist[index] < 0:
                sign *= -1
                if sublist[index] == -1:
                    pass    # -1 will be appended later because sign = -1
                else:
                    new_sublist.extend(prime_factors(-sublist[index]))
            elif sublist[index] == 0:
                zero_flag = 1
            else:
                if sublist[index] == 1:
                    new_sublist.extend([1])
                else:
                    new_sublist.extend(prime_factors(sublist[index]))
        if len(sublist) == 0:
            continue
        if sign == -1:
            new_sublist.append(-1)
        new_sublist.sort()
        if not zero_flag:
            lol_copy.append(new_sublist)
    # Now divide each sublist by divisor
    #print('lol_copy = ', lol_copy)
    divisorlist = prime_factors_list(divisorlist)
    for divisor in divisorlist:
        assert divisor > 0
        if divisor == 1:
            continue
        appended_lists = []  # Extra lists to add to lol_copy
        single_remainders = 0  # Take care of remainders for this divisor
        for sublist in lol_copy:
            if divisor in sublist:
                if len(sublist) > 1:
                    sublist.remove(divisor)
                else:
                    sublist[0] = 1
            elif len(sublist) == 0:
                continue
            elif sublist[-1] < 0:
                if (-sublist[-1] // divisor) != 0:
                    appended_lists.append([-(-sublist[-1] // divisor)])
                single_remainders -= -sublist[-1] % divisor
                sublist.pop()
            else:
                while len(sublist) > 1:
                    rem = sublist[-1] % divisor
                    sublist[-1] = sublist[-1] // divisor
                    if sublist[-1] != 0:
                        appended_lists.append(list(sublist))
                    sublist.pop()
                    sublist[-1] *= rem
                # Now sublist has length 1
                if sublist[0] <  0:
                    if (-sublist[0] // divisor) != 0:
                        appended_lists.append([-(-sublist[0] // divisor)])
                elif sublist[0] // divisor != 0:
                    appended_lists.append([sublist[0] // divisor])
                if sublist[0] < 0:
                    single_remainders -= -sublist[0] % divisor
                else:
                    single_remainders += sublist[0] % divisor
                sublist.pop()
        lol_copy.extend(appended_lists)
        assert not single_remainders % divisor
        if single_remainders:
            lol_copy.append([single_remainders // divisor])

        # Clean up - remove any empty lists, single digit lists and 1's from lists
        for ind1 in range(len(lol_copy)):
            if len(lol_copy[ind1]) == 0:
                continue
            all_ones = 1
            any_one = 0
            one_indices = []
            for ind in range(len(lol_copy[ind1])):
                if lol_copy[ind1][ind] != 1:
                    all_ones = 0
                else:
                    any_one = 1
                    one_indices.append(ind)
            if all_ones == 1:
                lol_copy[ind1] = [1]
            elif any_one == 1:
                for ind in reversed(one_indices):
                    lol_copy[ind1].pop(ind)
        remove_indices = []
        single_digits = 0
        for ind in range(len(lol_copy)):
            if lol_copy[ind] == []:
                remove_indices.append(ind)
            if len(lol_copy[ind]) == 1:
                remove_indices.append(ind)
                single_digits += lol_copy[ind][0]
        for ind in reversed(remove_indices):
            lol_copy.pop(ind)
        if single_digits:
            if single_digits == 1:
                lol_copy.append([single_digits])
            elif single_digits < 0:
                if single_digits == -1:
                    lol_copy.append([single_digits])
                else:
                    lol_copy.append([-1] + prime_factors(-single_digits))
            else:
                lol_copy.append(prime_factors(single_digits))
    return lol_copy


def prod_sum_eval(list_of_lists):
    """
    Evaluates sum of products of sublists
    :param list: list of lists of numbers
    :return: sum of product of sublist elements
    """
    assert isinstance(list_of_lists, list)
    total = 0
    for sublist in list_of_lists:
        assert isinstance(sublist, list)
        if len(sublist) > 0:
            prod = 1
            for item in sublist:
                prod *= item
            total += prod
    return total

"""
# Example
lol = [[1,7,8,5],[8,-10],[], [10, 1]]
print(prod_sum_divide(lol, [6]))
print('total = ', prod_sum_eval(prod_sum_divide(lol, [6])))
print('\n********************************************\n')
lol2 = [[9,10,11,16],[10,11,48,-1],[11,96,-1,1,-1], [96,-1,1]]
print(prod_sum_divide(lol2, [8,9,5]))
print('total = ', prod_sum_eval(prod_sum_divide(lol2, [8,9,5])))
print('\n********************************************\n')
lol3 = [[],[10,11,1,1],[11,14,1],[96,1,1]]
print(prod_sum_divide(lol3, [8,9,5]))
print('total = ', prod_sum_eval(prod_sum_divide(lol3, [8,9,5])))
print('\n********************************************\n')
lol4 = [[15,14,13,12,11,0], [15,14,13,12,7], [15,14,13,126], [15,14,1422], [15,10872], [46080]]
print(prod_sum_divide(lol4, [15,14,13,12,11]))
print('total = ', prod_sum_eval(prod_sum_divide(lol4, [15,14,13,12,11])))
#print(prod_sum_divide(lol4, [15,14,13,12,11,10])) # Throws an error since lol4 is not divisible by this
print('\n********************************************\n')
lol5 = [[2, 2, 5, 7, 11], [-1, 2, 3, 7, 11], [2, 2, 3, 7], [-1, 7]]
print(prod_sum_divide(lol5, [5]))
print('total = ', prod_sum_eval(prod_sum_divide(lol5, [5])))
print('\n********************************************\n')
lol6 = [[3, 7, 11, 11, 2], [3, 7, 11, 6], [3, 7, 6], [3, 4], [-1, 2, 2, 3, 7, 11, 2], [-1, 2, 2, 3, 7, 6], [-1, 2, 2, 3, 4], [-1, 2], [3, 7, 11, 2], [3, 7, 6], [3, 4], [-1, 2, 3, 7, 2], [-1, 2, 3], [-1, 2], [3, 2], [1]]
print('total before dividing: ', prod_sum_eval(lol6))
print(prod_sum_divide(lol6, [11]))
print('total = ', prod_sum_eval(prod_sum_divide(lol6, [11])))
"""
