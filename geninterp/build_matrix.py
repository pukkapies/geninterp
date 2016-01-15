from scipy.spatial import distance
import math
import numpy as np
import scipy.sparse as sparse

def gram_Wendland_MeshData(Int_obj):
    """
    Creates the Gram matrix from the Int_obj, using sparse representation
    from compact support of Wendland function
    Assumes that the Int_obj.gram_functions are zero for
    Int_obj.K.c * np.linalg.norm(i_pts[k,:] - j_pts[l,:]) >= 1
    :param Int_obj: Interpolation object
    :return: Gram matrix
    """
    # assert isinstance(data_obj, MeshData) before passing to this function
    # assert isinstance(Int_obj.K, Wendland) before passing to this function
    assert len(Int_obj.data_points)==1      #Temporary - future implementation will allow more data objects
    #TODO: Increase functionality to multiple MeshData objects
    # First calculate the range of indices the support covers in each dimension
    offset_ranges = [math.floor(1/(Int_obj.K.c * Int_obj.data_points[0].range_tup_list[i][2])) \
                     for i in range(len(Int_obj.data_points[0].range_tup_list))]
    print('offset_ranges = ', offset_ranges)
    sparse_gram = sparse.dok_matrix((Int_obj.data_points[0].numpoints, Int_obj.data_points[0].numpoints))
    numpoints_by_dim = Int_obj.data_points[0].numpoints_by_dim
    ndim = Int_obj.data_points[0].dim

    # Create an index matrix. Same shape as one of the self.grids but contains indices
    index_matrix = np.array(np.meshgrid(*[np.arange(Int_obj.data_points[0].numpoints_by_dim[d])
                                          for d in range(Int_obj.dim)], indexing='ij')).T
    # Enter elements by row
    i_unrolled = 0
    for i in index_matrix.reshape(Int_obj.data_points[0].numpoints, ndim):
        # i is an 1D array length ndim, giving the indices of a point in the dataset of the form [xi, yi, zi]
        # the x-ccord is changing the fastest, followed by y, etc.

        #i_unrolled = np.sum(np.array([i[d] * np.prod(numpoints_by_dim[:d]) for d in range(ndim)]))
        """
            # Break down i into indices along each dimension
            i_by_dim = []
            tempi = i
            for d in range(ndim):
                #i_by_dim[d] = i % numpoints_by_dim[d]
                i_by_dim.append(tempi % numpoints_by_dim[d])
                tempi -= i_by_dim[d]
                tempi //= numpoints_by_dim[d]
            """
        index_matrix_ranges = np.array(np.meshgrid(*[np.arange(max(0,-offset_ranges[d] + i[d]),
                                                      min(offset_ranges[d] + i[d] + 1, numpoints_by_dim[d]))
                                            for d in range(Int_obj.dim)])).T

        for j in index_matrix_ranges.reshape(index_matrix_ranges[...,0].size, ndim):
            # Compute only upper triangular elements
            j_unrolled = np.sum(np.array([j[d] * np.prod(numpoints_by_dim[:d]) for d in range(ndim)]))
            if i_unrolled <= j_unrolled:
                sparse_gram[i_unrolled, j_unrolled] = \
                        Int_obj.linfunc.gram_functions[0][0](Int_obj.data_points[0].grids[tuple(i[::-1])],
                                                             Int_obj.data_points[0].grids[tuple(j[::-1])])
            else:
                sparse_gram[i_unrolled, j_unrolled] = sparse_gram[j_unrolled, i_unrolled]

        i_unrolled += 1

    return sparse_gram


def gram_Wendland(Int_obj):
    """
    Function uses compact support of Wendland function to create the Gram matrix from the
    Int_obj.linfunc and Int_obj.data_points attributes
    Assumes that the Int_obj.gram_functions are zero for
    Int_obj.K.c * np.linalg.norm(i_pts[k,:] - j_pts[l,:]) >= 1
    No vectorisation
    :param Int_obj: Interpolation object
    :return: Gram matrix
    """
    # Use compact support to speed up Gram matrix population
    len_temp = len(Int_obj.linfunc.gram_functions) # Number of row/column blocks in Gram matrix

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
                    num_points_in_dataobj_i = Int_obj.data_points[i].numpoints
                    num_points_in_dataobj_j = Int_obj.data_points[j].numpoints
                    i_pts = Int_obj.data_points[i].points
                    j_pts = Int_obj.data_points[j].points
                    off_diag_block = np.zeros((num_points_in_dataobj_i, num_points_in_dataobj_j))
                    for k in range(num_points_in_dataobj_i):
                        for l in range(num_points_in_dataobj_j):
                            if Int_obj.K.c * np.linalg.norm(i_pts[k,:] - j_pts[l,:]) >= 1:
                                off_diag_block[k,l] = 0
                            else:
                                off_diag_block[k,l] = Int_obj.linfunc.gram_functions[i][j](i_pts[k,:], j_pts[l,:])
                    row.append(off_diag_block)

            list_of_rows.append(row)
    # Now calculate the diagonal blocks
    for i in range(len_temp):
        pts = Int_obj.data_points[i].points
        num_points_in_dataobj = Int_obj.data_points[i].numpoints
        diag_block = np.zeros((num_points_in_dataobj,num_points_in_dataobj))
        for j in range(num_points_in_dataobj):
            for k in range(num_points_in_dataobj):
                if k < j:
                    diag_block[j,k] = diag_block[k,j]
                elif Int_obj.K.c * np.linalg.norm(pts[j,:] - pts[k,:]) >= 1:
                    diag_block[j,k] = 0
                else:
                    diag_block[j,k] = Int_obj.linfunc.gram_functions[i][i](pts[j,:], pts[k,:])
        list_of_rows[i][i] = diag_block
    return np.vstack([np.hstack(list) for list in list_of_rows])



def gram(Int_obj):
    """
    Basic function to create Gram matrix from the Int_obj.linfunc and Int_obj.data_points attributes
    Uses vectorised cdist function
    :param Int_obj: Interpolation object
    :return: Gram matrix
    """

    len_temp = len(Int_obj.linfunc.gram_functions) # Number of rows/column blocks in Gram matrix

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
                row.append(distance.cdist(Int_obj.data_points[i].points,
                                          Int_obj.data_points[j].points,
                                          Int_obj.linfunc.gram_functions[i][j]))
        list_of_rows.append(row)
    # Now calculate the diagonal blocks
    for i in range(len_temp):
        pts = Int_obj.data_points[i].points
        num_points_in_dataobj = Int_obj.data_points[i].numpoints
        diag_block = np.zeros((num_points_in_dataobj,num_points_in_dataobj))
        for j in range(num_points_in_dataobj):
            for k in range(num_points_in_dataobj):
                if k < j:
                    diag_block[j,k] = diag_block[k,j]
                else:
                    diag_block[j,k] = Int_obj.linfunc.gram_functions[i][i](pts[j,:], pts[k,:])
        list_of_rows[i][i] = diag_block
    return np.vstack([np.hstack(list) for list in list_of_rows])



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
