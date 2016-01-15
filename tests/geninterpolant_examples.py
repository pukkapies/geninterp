import sys, os
current_path = os.getcwd()
current_path_list = current_path.split('/')
module_path_list = current_path_list[:-1]
module_path = '/'.join(module_path_list)

sys.path.append(module_path)
#sys.path.append(current_path)

from geninterp import *
import matplotlib.pyplot as plt

# Example
K = Gaussian(0.25, dimension=1)
#K = Wendland(13,5, dimension=1, c=0.45)
testpts = Data()
testpts.set_data(np.array([[0], [1], [2]]))
testpts.set_targets(np.array([1, 10, -1]))

testpts_derivs = Data()
testpts_derivs.set_data(np.array([[1], [2]]))
testpts_derivs.set_targets(np.array([0, 0]))

beta = np.hstack((testpts_derivs.targets, testpts.targets))

def f(x):
    return np.array([1])

basisfuns_simple = LinearFunctional()
basisfuns_simple.add_basis_function(K.eval)
basisfuns_simple.set_gram_function(0, 0, K.eval)

inter_simple = Interpolant(basisfuns_simple, [testpts], K)
inter_simple.solve_linear_system()

def s_simple(x): return inter_simple.eval(x)

print(s_simple(np.array([[0]])))
print(s_simple(np.array([[1]])))
print(s_simple(np.array([[2]])))
print(s_simple(np.array([[3]])))

##############################################################################################################

#K = Wendland(13,5, c=0.45)
K = Gaussian(1)

testpts = Data()
testpts.set_data(np.array([[0], [1], [2]]))
testpts.set_targets(np.array([1, 10, -1]))

testpts_derivs = Data()
testpts_derivs.set_data(np.array([[1], [2.5]]))
testpts_derivs.set_targets(np.array([0, 0]))

inter_orbderiv = OrbDerivInterpolant([testpts_derivs,testpts], K, f)
inter_orbderiv.solve_linear_system()

def s_orbderiv(x): return inter_orbderiv.eval(x)


print('++++++')
print(s_orbderiv(np.array([[0]])))
print(s_orbderiv(np.array([[1]])))
print(s_orbderiv(np.array([[2]])))
print(s_orbderiv(np.array([[3]])))


plt.figure()
xrange = np.linspace(-1, 5, 100).reshape(100,1)
plt.plot(xrange.reshape(100,), s_orbderiv(xrange))
plt.show()


##############################################################################################################

K = Gaussian(1,dimension=2)

basisfuns_simple = LinearFunctional()
basisfuns_simple.add_basis_function(K.eval)
basisfuns_simple.set_gram_function(0, 0, K.eval)

testpts = Data()
testpts.set_data(np.array([[0, 1], [1,1], [2,1]]))
testpts.set_targets(np.array([1, 10, -1]))

inter_simple = Interpolant(basisfuns_simple, [testpts], K)
inter_simple.solve_linear_system()

def s_simple(x): return inter_simple.eval(x)

print('++++++')
print(s_simple(np.array([[0.5,1]])))
print(s_simple(np.array([[1,1]])))
print(s_simple(np.array([[2,1]])))


