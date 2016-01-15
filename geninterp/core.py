__author__ = 'Kevin Webster'

from .kernel import *
from .build_matrix import *
import scipy.sparse as sparse
from scipy.sparse.linalg.dsolve import spsolve
import inspect, time

# For printing out numpy arrays without line breaks
np.set_printoptions(linewidth=np.nan)

class Data(object):

    def __init__(self):
        """
        Object contains points as a 2D numpy array, and targets as 1D numpy array.
        Input points with set_data method, and targets with set_targets
        :return:
        """
        self.dim = None     # Phase space dimension - initialises when data is entered
        self.numpoints = 0  # Number of data points entered
        self.points = np.array([[]])    # Always 2D np.array
        self.targets = np.array([])     # Always 1D np.array

    def set_data(self, pts):
        """
        Stores sets of data points in self.points and checks the format
        :param pts: A set of points. Must be 2D np.array with points stacked along rows
        :return: None
        """
        assert isinstance(pts, np.ndarray)
        assert pts.ndim == 2
        self.dim = pts.shape[1] # Phase space dimension
        if len(self.targets) == 0:
            self.numpoints = pts.shape[0]
        else:
            assert self.numpoints == pts.shape[0]
        self.points = pts

    def set_targets(self, tgts):
        """
        Sets self.targets
        Stores sets of data points and checks the format
        :param tgts: A set of target values as 1D np.array
        :return: None
        """
        assert isinstance(tgts, np.ndarray)
        assert tgts.ndim == 1
        if self.points.size == 0:
            self.numpoints = len(tgts)
        else:
            assert len(tgts) == self.numpoints
        self.targets = tgts


class MeshData(Data):
    """
    MeshData class useful for fast Wendland kernel implementations, whenever the data points are
    defined by a simple grid
    self.points stores the points in a 2D np.array with points in rows, such that the first variable
    changes the fastest as you read down the array
    """
    def __init__(self, *min_max_tup, function=None):
        """
        Initialiser
        :param min_max_tup: sequence of 3-tuples to define ranges, in the form (xmin, xmax, xstep)
        :param function: targets are equal to the function applied to each data point. When set,
                        ensure that it supports broadcasting, and it broadcasts along the last dimension.
                        If it is not set, targets must be entered manually
        """
        super().__init__()
        self.dim = len(min_max_tup)
        self.range_tup_list = list(min_max_tup)
        self.range_list = [np.arange(r[0],r[1],r[2]) for r in min_max_tup]
        # eg self.grids will have shape (zdim, ydim, xdim, self.dim) in three dimensions
        self.grids = np.array(np.meshgrid(*self.range_list, indexing='ij')).T

        self.numpoints = self.grids[...,0].size
        self.numpoints_by_dim = np.array([len(nparange) for nparange in self.range_list])

        # NB in self.points, the x coord will change fastest, followed by y, etc.
        self.set_data(self.grids.reshape(self.numpoints, self.dim))
        if function != None:
            targets = function(self.grids)
            self.set_targets(targets.reshape(self.numpoints))


class LinearFunctional(object):
    """
    Stores a list of functions of two variables that are to be used as basis functions in the
    interpolant expansion. NOTE: these should be defined as functions applied
    to first argument of the kernel, before evaluation. (So they are not yet functionals.)
    Also stores the functionals applied to both arguments of the kernel, to be used
    in the Gram matrix construction.
    NB: Add basis functions first, then gram functions!
    """
    def __init__(self):
        self.basis_functions = []   # Store basis functions in a list
        self.gram_functions = []  # Store Gram functions in a list of lists

    def add_basis_function(self, function):
        self.basis_functions.append(function)
        # Add zero column and zero row
        len_temp = len(self.gram_functions)
        for i in range(len_temp):
            self.gram_functions[i].append(False)
        self.gram_functions.append((len_temp + 1) * [False])

    def get_basis_function_index(self, function):
        return self.basis_functions.index(function)

    # def swap_basis_function_indices(self, function1, function2):
    #     temp1, temp2 = self.basis_functions.index(function1), self.basis_functions.index(function2)
    #     self.basis_functions[temp1], self.basis_functions[temp2] = self.basis_functions[temp2], self.basis_functions[temp1]

    def remove_basis_function_index(self, index):
        self.basis_functions.pop(index)
        len_temp = len(self.gram_functions)
        for i in range(len_temp):
            self.gram_functions[i].pop(index)
        self.gram_functions.pop(index)

    def remove_basis_function(self, function):
        index = self.get_basis_function_index(function)
        self.remove_basis_function_index(index)
        #self.basis_functions.remove(function)

    def set_gram_function(self, i, j, function):
        assert (i < len(self.gram_functions)) and (j < len(self.gram_functions))
        self.gram_functions[i][j] = function
        if self.gram_functions[j][i] == False and j != i:
            self.gram_functions[j][i] = lambda x, y: self.gram_functions[i][j](y,x)


class Interpolant(object):

    def __init__(self, linearfunctional, data_list, K):
        """
        Initialiser for orbital derivative interpolant
        :param linearfunctional: LinearFunctional object. Number of basis functions should be the same as
                                the number of Data objects. gram_functions attribute should contain
                                functions to populate the Gramian matrix, arranged in a list of rows,
                                with a function for each Data block (for each combination of linear functionals)
        :param data_list: list of Data objects, should have same length as basis functions,
                    one (but not all) Data objects could be empty
        :return: Interpolant object
        Example usage:
            K = Gaussian(1,dimension=1)

            basisfuns_simple = LinearFunctional()
            basisfuns_simple.add_basis_function(K.eval)
            basisfuns_simple.set_gram_function(0, 0, K.eval)

            testpts = Data()
            testpts.set_data(np.array([[0], [1], [2]]))
            testpts.set_targets(np.array([1, 10, -1]))

            inter_simple = Interpolant(basisfuns_simple, [testpts])
            inter_simple.solve_linear_system()

            def s_simple(x): return inter_simple.eval(x)

            print(s_simple(np.array([[0]])))
            print(s_simple(np.array([[1]])))
            print(s_simple(np.array([[2]])))
        """
        assert isinstance(linearfunctional, LinearFunctional)
        self.dim = None
        for data_obj in data_list:
            assert isinstance(data_obj, Data)
            if data_obj.numpoints != 0: # data_obj is not empty
                if self.dim == None:    # Dimension not yet initialised
                    self.dim = data_obj.dim
                if data_obj.dim != self.dim:
                    print("ERROR initialising Interpolant - data objects have different dimension")
                    exit(1)
        self.data_points = data_list   # List of Data objects, one object for each different basis
        self.coefficients = None # 1D np.array containing coefficients in function expansion
        # Check that number of Data objects and basis functions match
        if len(data_list) != len(linearfunctional.basis_functions):
            print('data_points = ', data_list)
            print('self.linfunc.basis_functions = ', linearfunctional.basis_functions)
            print("ERROR: Incompatible lengths of Interpolant.data_points and Interpolant.basis.functionals")
            exit(1)
        self.linfunc = linearfunctional # LinearFunctional object - functions used in expansion terms
        assert isinstance(K, Kernel)
        self.K = K
        self.K.dim = self.dim
        self.regularisation = 0     #regularisation parameter
        self.gram = None    # Interpolation (Gramian) matrix to be constructed from linearfunctional
        self.beta = np.array([]) # 1D np.array to contain target values contained in data_list

    def set_lambda(self, l):
        """
        Sets regularisation parameter (default is zero)
        :return: None
        """
        assert l >= 0
        self.regularisation = l

    def test_new_gram(self):
        self._check_for_empty_Data()
        start = time.time()
        print('making sparse gram')
        self.sparse_gram = sparse.csr_matrix(gram_Wendland_MeshData(self))
        print('finished sparse gram')
        end_sparse = time.time()
        print('making Gram with gram() function')
        self.gram = gram(self)
        print('finished gram')
        end_gram = time.time()
        print('making Gram with gram_Wendland function')
        self.gram_Wendland = gram_Wendland(self)
        print('finished gram_Wendland')
        end_gram_Wendland = time.time()

        print('Time for sparse: ', end_sparse - start)
        print('Time for gram: ', end_gram - end_sparse)
        print('Time for gram_Wendland: ', end_gram_Wendland - end_gram)

        print('type(self.sparse_gram) = ', type(self.sparse_gram))

        print('Difference in norm: ', np.linalg.norm(self.gram - self.sparse_gram.toarray()))

        for data_object in self.data_points:
            #self.dim = data_object.points.shape[0]
            self.beta = np.hstack((self.beta, data_object.targets))

        self.coefficients = np.linalg.solve(self.gram, self.beta)
        self.coefficients2 = spsolve(self.sparse_gram, self.beta)

    def solve_linear_system(self, A=None, use_Wendland_compsupp=True, regularisation=None):
        """
        Solves the linear system defined by generalised interpolation problem
        :param A: Interpolation matrix
        :param use_Wendland_compsupp: When A is not given and the kernel is Wendland kernel, use the
        compact support to speed up population of interpolation matrix.
        :return: solution vector stored in self.coefficients
        """
        if regularisation!=None:
            self.regularisation=regularisation

        # Check the Data list is not empty
        data_empty = True
        for data_object in self.data_points:
            if data_object.numpoints > 0: data_empty = False
        if data_empty == True:
            print("ERROR: Cannot solve linear system - no data points")
            exit(1)
        self._check_for_empty_Data()
        if A==None:
            print('making Gram matrix')
            if use_Wendland_compsupp==False or not isinstance(self.K, Wendland):
                print('using gram')
                self.gram = gram(self)
            else:
                print('using gram_Wendland')
                self.gram = gram_Wendland(self)
            print('finished making Gram matrix')
        else:
            self._check_gram(A)
            self.gram = A
        if self.regularisation != 0:
            self.gram = self.gram + (self.regularisation * sparse.identity(self.gram.shape[0]))
        print('solving linear system')
        for data_object in self.data_points:
            #self.dim = data_object.points.shape[0]
            self.beta = np.hstack((self.beta, data_object.targets))
        if sparse.issparse(self.gram):
            if not isinstance(self.gram, sparse.csr.csr_matrix):
                self.gram = sparse.csr_matrix(self.gram)
            self.coefficients = spsolve(self.gram, self.beta)
        else:
            self.coefficients = np.linalg.solve(self.gram, self.beta)
        print('finished linear system')

    def eval(self, y):
        """
        Evaluates interpolant at y. The data point (points which are centres
        of basis functions) is evaluated in the first argument of
        basis_function, so y is the second argument.
        This is important, as basis functions might not be symmetric in their arguments.
        #Allows y to be an array of points (arranged in columns)
        :param y: np.array with y.shape[0] = self.dim
        :return: np.array which is evaluation of interpolant at y
        """
        if not isinstance(y, np.ndarray):
            print("ERROR: argument passed to Interpolant.eval is not np.ndarray")
            exit(1)
        elif y.shape[-1] != self.dim:
            print("ERROR: argument passed to Interpolant.eval is incorrect dimension")
            exit(1)

        #num_points = y.shape[0]
        # output = np.zeros(num_points)
        # for i in range(num_points):
        #     point_counter = 0
        #     for index, data in enumerate(self.data_points):
        #         for j in range(self.data_points[index].points.shape[1]):
        #             output[i] += self.coefficients[point_counter] * self.linfunc.basis_functions[index](
        #                     self.data_points[index].points[:, j], y[:, i])
        #             point_counter += 1
        # return output

        output = 0
        point_counter = 0
        for index, data in enumerate(self.data_points):
            for j in range(self.data_points[index].numpoints):
                output += self.coefficients[point_counter] * self.linfunc.basis_functions[index](
                        self.data_points[index].points[j, :], y)
                point_counter += 1

        return output

    ####################################### PRIVATE FUNCTIONS #######################################

    def _check_for_empty_Data(self):
        """
        Checks through the self.data_points list to see if any Data objects are empty. If there are
        empty objects, they are removed, along with the corresponding basis and gram functions
        :return:
        """
        empty_indices = [] # Records which Data objects are empty, if any

        for data_obj in self.data_points:
            if data_obj.numpoints == 0:
                empty_indices.append(self.data_points.index(data_obj))
        if len(empty_indices) == len(self.data_points):
            print("ERROR: no data in self.data_points")
            exit(1)
        for index in reversed(empty_indices):
            self.data_points.pop(index)
            self.linfunc.remove_basis_function_index(index) # Also removes Gram function

    def _check_gram(self, A):
        """
        Checks that a user-input Gram matrix is right size
        :return: None
        """
        sparse_module_classes = tuple(x[1] for x in inspect.getmembers(sparse,inspect.isclass))
        assert isinstance(A, sparse_module_classes + (np.ndarray, np.matrixlib.defmatrix.matrix))
        assert A.ndim == 2
        num_data_points = sum([self.data_points[i].numpoints for i in range(len(self.data_points))])
        assert A.shape[0] == num_data_points
        assert A.shape[1] == num_data_points


class OrbDerivInterpolant(Interpolant):

    def __init__(self, data_list, K, f):
        """
        Initialiser for orbital derivative interpolant
        :param linearfunctional: LinearFunctional object
        :param data_list: list of two Data objects, first for orbital derivative targets, second for
                        function value targets. If only orbital derivative is required, enter an empty Data object
                        into the second argument
        :param K: Kernel object
        :param f: RHS of vector field - function of 1D (length of dimension equal to Data objects) np.array returns
                    a 1D np.array of same length
        :return: Interpolant object
        Example usage:
            K = Wendland(3,2)

            testpts = Data()
            testpts.set_data(np.array([[0], [1], [2]]))
            testpts.set_targets(np.array([1, 10, -1]))

            testpts_derivs = Data()
            testpts_derivs.set_data(np.array([[1], [2]]))
            testpts_derivs.set_targets(np.array([0, 0]))

            def f(x): return np.array([1])

            inter_orbderiv = OrbDerivInterpolant([testpts_derivs,testpts], K, f)
            inter_orbderiv.solve_linear_system()

            def s_orbderiv(x): return inter_orbderiv.eval(x)

            print(s_orbderiv(np.array([[0]])))
            print(s_orbderiv(np.array([[1]])))
            print(s_orbderiv(np.array([[2]])))

            plt.figure()
            xrange = np.linspace(-1, 5, 100).reshape(1,100)
            plt.plot(xrange.reshape(100,), s_gen(xrange))
            plt.show()
        """
        def orbder_one_arg(x, y):
            return K.psi1(x, y) * np.sum((x - y) * f(x),axis=-1)

        def orbder_two_args(x, y):
            return K.psi2(y, x) * np.sum((x - y) * f(x),axis=-1) * np.sum((y - x) * f(y),axis=-1) - \
                             K.psi1(y, x) * np.sum(f(y) * f(x),axis=-1)

        linfunc = LinearFunctional()
        if isinstance(K, RBF):
            linfunc.add_basis_function(orbder_one_arg)
            linfunc.add_basis_function(K.eval)
            linfunc.set_gram_function(0,0,orbder_two_args)
            linfunc.set_gram_function(0,1,orbder_one_arg)
            linfunc.set_gram_function(1,1, function = K.eval)
        else:
            print("WARNING: Basis and gram functions need to be set for OrbDerivInterpolant.linearfunctional")
        super().__init__(linfunc, data_list, K)
        self.f = f
        # attributes for function value interpolation
        self.val_points = np.array([])
        self.num_val_points = 0
        self.val_targets = np.array([])
        # attributes for orbital derivative interpolation
        self.orb_deriv_points = np.array([])
        self.num_orb_deriv_points = 0
        self.orb_deriv_targets = np.array([])

    def _gram(self):

        if isinstance(self.K, Wendland):
            return super()._gram_Wendland()

        else:
            return super()._gram()
