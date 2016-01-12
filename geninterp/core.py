__author__ = 'Kevin Webster'

from geninterp.kernel import *
from scipy.spatial import distance
import scipy.sparse as sparse
from scipy.sparse.linalg.dsolve import spsolve

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

    def shape(self):
        return self.points.shape

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

# Function not used any more
# def gram(x_obj, y_obj, function):
#         """
#         Returns the Gramian matrix for function k which takes 2 arguments,
#         evaluated at set of points x. Array x should be arranged with
#         points in columns
#         :param x_obj: Data object containing points to construct Gramian matrix along
#                       the first dimension (changes on different rows)
#         :param y_obj: Data object containing points to construct Gramian matrix along
#                       the second dimension (changes on different columns)
#         :param function: function of two arguments with which to construct
#                         Gramian matrix.
#                         NB function need not be a kernel, in which case this isn't
#                         actually a Gram matrix... but it's useful for constructing
#                         off-diagonal blocks in an interpolation matrix (which altogether
#                         is a Gram matrix)
#         :return: Gram matrix as a 2D numpy array
#         """
#         assert isinstance(x_obj, Data) and isinstance(y_obj, Data)
#         assert x_obj.shape()[0] != 0 and y_obj.shape()[0] != 0 # Checks the Data objects are not empty
#         x = x_obj.points
#         y = y_obj.points
#         num_xpoints = x.shape[0]
#         num_ypoints = y.shape[0]
#         gram = np.zeros((num_xpoints, num_ypoints))
#         for i in range(num_xpoints):
#             for j in range(num_ypoints):
#                 gram[i, j] = function(x[:, i], y[:, j])
#         return gram


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
        self.gram = None    # Interpolation (Gramian) matrix to be constructed from linearfunctional
        self.beta = np.array([]) # 1D np.array to contain target values contained in data_list

    def solve_linear_system(self, A=None, use_Wendland_compsupp=False):
        """
        Solves the linear system defined by generalised interpolation problem
        :param A: Interpolation matrix
        :param use_Wendland_compsupp: When A is not given and the kernel is Wendland kernel, use the
        compact support to speed up population of interpolation matrix.
        :return: solution vector stored in self.coefficients
        """
        # Check the Data list is not empty
        data_empty = True
        for data_object in self.data_points:
            if data_object.numpoints > 0: data_empty = False
        if data_empty == True:
            print("ERROR: Cannot solve linear system - no data points")
            exit(1)
        if A==None:
            print('making Gram matrix')
            if use_Wendland_compsupp==False:
                self._gram()
            else:
                self._gram_Wendland()
            print('finished making Gram matrix')
        else:
            self.gram = A
        print('solving linear system')
        for data_object in self.data_points:
            #self.dim = data_object.points.shape[0]
            self.beta = np.hstack((self.beta, data_object.targets))
        if isinstance(self.gram, sparse.csr.csr_matrix):
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

    def _gram_Wendland(self):
        """
        Creates the Gram matrix from the self.linfunc and self.data_points attributes
        Checks if any of the Data objects are empty, if so removes them from the matrix
        and data_points and linfunc attributes
        Uses compact support of Wendland function, no vectorisation
        :return: Gram matrix
        """
        self._check_for_empty_Data()
        # Use compact support to speed up Gram matrix population
        len_temp = len(self.linfunc.gram_functions) # Number of rows/column blocks in Gram matrix

        # Using a for loop over the off-diagonal blocks of the Gram matrix, using the symmetry of the matrix
        # Then calculate the diagonal blocks, again using symmetry
        list_of_rows = []
        if len_temp > 0:
            for i in range(len_temp):
                row = []
                for j in range(len_temp):
                    if j == i:
                        row.append(np.array([0])) # Dummy value, to be replaced in next loop
                    elif j < i:
                        row.append(list_of_rows[j][i].T)
                    else:
                        num_points_in_dataobj_i = self.data_points[i].numpoints
                        num_points_in_dataobj_j = self.data_points[j].numpoints
                        i_pts = self.data_points[i].points
                        j_pts = self.data_points[j].points
                        off_diag_block = np.zeros((num_points_in_dataobj_i, num_points_in_dataobj_j))
                        for k in range(num_points_in_dataobj_i):
                            for l in range(num_points_in_dataobj_j):
                                if self.K.c * np.linalg.norm(i_pts[k,:] - j_pts[l,:]) >= 1:
                                    off_diag_block[k,l] = 0
                                else:
                                    off_diag_block[k,l] = self.linfunc.gram_functions[i][j](i_pts[k,:], j_pts[l,:])
                        row.append(off_diag_block)

                list_of_rows.append(row)
        # Now calculate the diagonal blocks
        for i in range(len_temp):
            pts = self.data_points[i].points
            num_points_in_dataobj = self.data_points[i].numpoints
            print('number of data points = ', num_points_in_dataobj)
            diag_block = np.zeros((num_points_in_dataobj,num_points_in_dataobj))
            for j in range(num_points_in_dataobj):
                print('j = ', j)
                for k in range(num_points_in_dataobj):
                    if k < j:
                        diag_block[j,k] = diag_block[k,j]
                    elif self.K.c * np.linalg.norm(pts[j,:] - pts[k,:]) >= 1:
                        diag_block[j,k] = 0
                    else:
                        diag_block[j,k] = self.linfunc.gram_functions[i][i](pts[j,:], pts[k,:])
            list_of_rows[i][i] = diag_block
        self.gram = np.vstack([np.hstack(list) for list in list_of_rows])

    def _gram(self):
        """
        Creates the Gram matrix from the self.linfunc and self.data_points attributes
        Checks if any of the Data objects are empty, if so removes them from the matrix
        and data_points and linfunc attributes
        Uses vectorised cdist function
        :return: Gram matrix
        """

        self._check_for_empty_Data()

        len_temp = len(self.linfunc.gram_functions) # Number of rows/column blocks in Gram matrix

        # Using a for loop over the off-diagonal blocks of the Gram matrix, using the symmetry of the matrix
        # Then calculate the diagonal blocks, again using symmetry
        list_of_rows = []
        for i in range(len_temp):
            row = []
            for j in range(len_temp):
                if j == i:
                    row.append(np.array([0])) # Dummy value, to be replaced in next loop
                elif j < i:
                    row.append(list_of_rows[j][i].T)
                else:
                    row.append(distance.cdist(self.data_points[i].points,
                                              self.data_points[j].points,
                                              self.linfunc.gram_functions[i][j]))
            list_of_rows.append(row)
        # Now calculate the diagonal blocks
        for i in range(len_temp):
            pts = self.data_points[i].points
            num_points_in_dataobj = self.data_points[i].numpoints
            diag_block = np.zeros((num_points_in_dataobj,num_points_in_dataobj))
            for j in range(num_points_in_dataobj):
                for k in range(num_points_in_dataobj):
                    if k < j:
                        diag_block[j,k] = diag_block[k,j]
                    else:
                        diag_block[j,k] = self.linfunc.gram_functions[i][i](pts[j,:], pts[k,:])
            list_of_rows[i][i] = diag_block
        self.gram = np.vstack([np.hstack(list) for list in list_of_rows])


        # Using a for loop over the off-diagonal blocks of the Gram matrix, using the symmetry of the matrix
        """
        list_of_rows = []
        for i in range(len_temp):
            row = []
            for j in range(len_temp):
                if j < i:
                    row.append(list_of_rows[j][i].T)
                else:
                    row.append(distance.cdist(self.data_points[i].points,
                                              self.data_points[j].points,
                                              self.linfunc.gram_functions[i][j]))
            list_of_rows.append(row)
        self.gram = np.vstack([np.hstack(list) for list in list_of_rows])
        """

        # Vectorise, using distance.cdist function
        """
        self.gram = np.vstack(tuple(np.hstack(tuple(distance.cdist(self.data_points[i].points,
                                                                         self.data_points[j].points,
                                                                         self.linfunc.gram_functions[i][j])
                                                          for j in range(len_temp)))) for i in range(len_temp))
        """

        # Vectorise, using gram function
        """
        self.gram = np.vstack(tuple(np.hstack(tuple(gram(self.data_points[i], self.data_points[j], \
                                                         self.linfunc.gram_functions[i][j]) for j in range(len_temp)))) \
                              for i in range(len_temp))
        """

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
        super().__init__(linfunc, data_list)
        self.K = K
        self.K.dim = self.dim   # Set the kernel dimension
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
