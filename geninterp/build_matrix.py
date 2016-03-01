from scipy.spatial import distance
import math
import numpy as np
import os
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

###############################################################################################################

def complete_gram_Wendland(Int_obj, filepath):
    """
    Function completes a previously interrupted build of the Gram matrix using gram_Wendland function.
    Int_obj needs to have a gramblockdict attribute: a dict with keys equal to block indices,
    and corresponding values equal to 2D array. Each 2D array might not be complete.
    :param Int_obj: Interpolation object
    :param filepath: path to the appropriate '../Gram matrix' folder
    :return: Gram matrix
    """
    len_temp = len(Int_obj.linfunc.gram_functions) # Number of row/column blocks in Gram matrix

    # Check for incomplete blocks
    list_of_rows = []
    if len_temp > 0:
        for i in range(len_temp):
            row = []
            for j in range(len_temp):
                if j == i:
                    row.append(np.array([0])) # Dummy value, to be replaced in next loop
                elif j < i:
                    row.append(list_of_rows[j][i].T)
                    if True:
                        gramfileblock_path = filepath + '/Block' + str(i) + '-' + str(j)
                        if not os.path.exists(gramfileblock_path): os.makedirs(gramfileblock_path)
                        np.save(gramfileblock_path, list_of_rows[j][i].T)
                else:
                    blockpath = filepath + 'Block' + str(i) + '-' + str(j)
                    if os.path.exists(blockpath):
                        num_points_in_dataobj_i = Int_obj.data_points[i].numpoints  #Number of rows in block
                        num_points_in_dataobj_j = Int_obj.data_points[j].numpoints  #Number of cols in block
                        i_pts = Int_obj.data_points[i].points
                        j_pts = Int_obj.data_points[j].points
                        blockcheckflag = 0
                        for k in range(len(num_points_in_dataobj_i)):
                            if 'row' + str(k) + '.npy' not in os.listdir(blockpath + '/'):
                                if not blockcheckflag:  #First time we found a row not in the directory
                                    print('Completing Gram block (%d,%d) from row %d...' % (i,j,k))

                                    off_diag_block = np.zeros((num_points_in_dataobj_i, num_points_in_dataobj_j))
                                    rowlisttemp = [np.load(blockpath + '/row' + str(rownumber) + '.npy')
                                                   for rownumber in range(k)]
                                    off_diag_block[0:k,:] = np.vstack(rowlisttemp)
                                    blockcheckflag = 1
                                for l in range(num_points_in_dataobj_j):
                                    if Int_obj.K.c * np.linalg.norm(i_pts[k,:] - j_pts[l,:]) >= 1:
                                        off_diag_block[k,l] = 0
                                    else:
                                        off_diag_block[k,l] = Int_obj.linfunc.gram_functions[i][j](i_pts[k,:], j_pts[l,:])
                                if True:
                                    gramfileblockrow_path = blockpath + '/'
                                    if not os.path.exists(gramfileblockrow_path): os.makedirs(gramfileblockrow_path)
                                    np.save(gramfileblockrow_path + 'row' + str(k), off_diag_block[k,:])
                            row.append(off_diag_block)
                    else:       #Block folder doesn't exist yet - so calculate from scratch
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
                            if True:
                                gramfileblockrow_path = filepath + '/Block' + str(i) + '-' + str(j) + '/'
                                print('gramfileblockrow_path = ', gramfileblockrow_path)
                                if not os.path.exists(gramfileblockrow_path):
                                    os.makedirs(gramfileblockrow_path)
                                np.save(gramfileblockrow_path + 'row' + str(k), off_diag_block[k,:])
                        row.append(off_diag_block)
                        if True:
                            gramfileblock_path = filepath + '/Block' + str(i) + '-' + str(j)
                            if not os.path.exists(gramfileblock_path): os.makedirs(gramfileblock_path)
                            np.save(gramfileblock_path, off_diag_block)

            list_of_rows.append(row)

    # Now calculate the diagonal blocks
    for i in range(len_temp):
        pts = Int_obj.data_points[i].points
        num_points_in_dataobj = Int_obj.data_points[i].numpoints
        blockpath = filepath + '/Block' + str(i) + '-' + str(i)
        if os.path.exists(blockpath):
            blockcheckflag = 0
            for j in range(num_points_in_dataobj):
                if 'row' + str(j) + '.npy' not in os.listdir(blockpath + '/'):
                    if not blockcheckflag:  #First time we found a row not in the directory
                        print('Completing Gram block (%d,%d) from row %d...' % (i,i,j))
                        diag_block = np.zeros((num_points_in_dataobj,num_points_in_dataobj))
                        rowlisttemp = [np.load(blockpath + '/row' + str(rownumber) + '.npy')
                                       for rownumber in range(j)]
                        diag_block[0:j,:] = np.vstack(rowlisttemp)
                        blockcheckflag = 1
                    for k in range(num_points_in_dataobj):
                        if k < j:
                            diag_block[j,k] = diag_block[k,j]
                        elif Int_obj.K.c * np.linalg.norm(pts[j,:] - pts[k,:]) >= 1:
                            diag_block[j,k] = 0
                        else:
                            diag_block[j,k] = Int_obj.linfunc.gram_functions[i][i](pts[j,:], pts[k,:])
                    if True:
                        gramfileblockrow_path = filepath + '/Block' + str(i) + '-' + str(i) + '/'
                        if not os.path.exists(gramfileblockrow_path): os.makedirs(gramfileblockrow_path)
                        np.save(gramfileblockrow_path +'row' +str(j), diag_block[j,:])
            list_of_rows[i][i] = diag_block
            if True:
                gramfileblock_path = filepath + '/Block' + str(i) + '-' + str(i)
                if not os.path.exists(gramfileblock_path): os.makedirs(gramfileblock_path)
                np.save(gramfileblock_path, list_of_rows[i][i])
        else:   # Block folder doesn't exist yet
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
                if True:
                    gramfileblockrow_path = filepath + '/Block' + str(i) + '-' + str(i) + '/'
                    if not os.path.exists(gramfileblockrow_path): os.makedirs(gramfileblockrow_path)
                    np.save(gramfileblockrow_path +'row' +str(j), diag_block[j,:])
            list_of_rows[i][i] = diag_block
            if True:
                gramfileblock_path = filepath + '/Block' + str(i) + '-' + str(i)
                if not os.path.exists(gramfileblock_path): os.makedirs(gramfileblock_path)
                np.save(gramfileblock_path, list_of_rows[i][i])

    return np.vstack([np.hstack(list) for list in list_of_rows])

###############################################################################################################

def gram_Wendland_sparse(Int_obj, filepath, last_saved_row=(0,0,0)):
    """
    Function computes and stores a sparse representation of the Gram matrix. The data for each row is stored,
    along with the row indices where the data appears. Upon completion the indexptr list is constructed and the
    sparse csr matrix is returned.
    Also completes a previously interrupted build.
    Int_obj needs to have a gramblockdict attribute: a dict with keys equal to block indices,
    and corresponding values equal to 2D array. Each 2D array might not be complete.
    :param Int_obj: Interpolation object
    :param filepath: full path to the appropriate '../Gram matrix' folder
    :param last_saved_row: Option to provide (Block_i, Block_j, lastsavedrow) number, to save the function from
            searching through all the saved rows in the folder
    :return: Gram matrix
    """
    len_temp = len(Int_obj.linfunc.gram_functions) # Number of row/column blocks in Gram matrix

    block_i = last_saved_row[0]
    block_j = last_saved_row[1]
    lastrow = last_saved_row[2]
    for i in range(block_i, len_temp):
        for j in range(block_j, len_temp):
            blockpath = filepath + '/Block' + str(i) + '-' + str(j) + '/'
            blockpathexists = 0
            if os.path.exists(blockpath):
                blockpathexists = 1
                print('Found existing Data directory. Looking for last computed row...')
            elif j >= i:
                os.mkdir(blockpath)
                #blockpathexists = 1
            # Now calculate blocks - start with diagonal blocks
            if j == i:  #Construct coo matrix. Save row indices for upper triangular part (including diagonal)
                num_points_in_dataobj = Int_obj.data_points[i].numpoints
                pts = Int_obj.data_points[i].points
                blockcheckflag = 0
                for k in range(lastrow, num_points_in_dataobj):
                    rowindices = np.array([], dtype=np.uint16)   # to store row indices of data (upper triangular only)
                    rowdata = np.array([])      # to store values of nonzero row elements (upper triangular part only)
                    if True: #blockpathexists # check which rows have been completed
                        if ('row' + str(k) + 'data.npy' not in os.listdir(blockpath)) or \
                                ('row' + str(k) + 'indices.npy' not in os.listdir(blockpath)):
                            # This row wasn't completed - either data file or row index file is incomplete
                            if (not blockcheckflag) and blockpathexists: #First time we found a row not in the directory
                                print('Completing Gram block (%d,%d) from row %d...' % (i,j,k))
                                blockcheckflag = 1
                            for l in range(k, num_points_in_dataobj):   # start from the diagonal element
                                if Int_obj.K.c * np.linalg.norm(pts[k,:] - pts[l,:]) < 1:
                                    rowindices = np.append(rowindices, l)
                                    rowdata = np.append(rowdata, Int_obj.linfunc.gram_functions[i][j](pts[k,:],
                                                                                                      pts[l,:]))
                            if True:
                                np.save(blockpath + 'row' + str(k) + 'data', rowdata)
                                np.save(blockpath + 'row' + str(k) + 'indices', rowindices)
                    """
                    # For some reason the code is a lot faster when it always executes the previous block
                    else:       # Block folder doesn't exist yet - no need to check for completed rows
                        for l in range(k, num_points_in_dataobj):
                            if Int_obj.K.c * np.linalg.norm(pts[k,:] - pts[l,:]) < 1:
                                rowindices = np.append(rowindices, l)
                                rowdata = np.append(rowdata, Int_obj.linfunc.gram_functions[i][j](pts[k,:],
                                                                                                  pts[l,:]))
                            if True:
                                np.save(blockpath + 'row' + str(k) + 'data', rowdata)
                                np.save(blockpath + 'row' + str(k) + 'indices', rowindices)
                    """
                lastrow=0

            elif j > i:   # construct upper diagonal blocks as csr matrices
                num_points_in_dataobj_i = Int_obj.data_points[i].numpoints  #Number of rows in block
                num_points_in_dataobj_j = Int_obj.data_points[j].numpoints  #Number of cols in block
                i_pts = Int_obj.data_points[i].points
                j_pts = Int_obj.data_points[j].points
                blockcheckflag = 0
                for k in range(lastrow, num_points_in_dataobj_i):
                    rowindices = np.array([], dtype=np.uint16)   # to store row indices of data
                    rowdata = np.array([])      # to store values of nonzero row elements
                    if blockpathexists: # check which rows have been completed
                        if ('row' + str(k) + 'data.npy' not in os.listdir(blockpath)) or \
                                ('row' + str(k) + 'indices.npy' not in os.listdir(blockpath)):
                            # This row wasn't completed - either data file or row index file is incomplete
                            if not blockcheckflag:  #First time we found a row not in the directory
                                print('Completing Gram block (%d,%d) from row %d...' % (i,j,k))
                                blockcheckflag = 1
                            for l in range(num_points_in_dataobj_j):
                                if Int_obj.K.c * np.linalg.norm(i_pts[k,:] - j_pts[l,:]) < 1:
                                    rowindices = np.append(rowindices, l)
                                    rowdata = np.append(rowdata, Int_obj.linfunc.gram_functions[i][j](i_pts[k,:],
                                                                                                      j_pts[l,:]))
                            if True:
                                np.save(blockpath + 'row' + str(k) + 'data', rowdata)
                                np.save(blockpath + 'row' + str(k) + 'indices', rowindices)
                    else:       # Block folder doesn't exist yet - no need to check for completed rows
                        for l in range(num_points_in_dataobj_j):
                            if Int_obj.K.c * np.linalg.norm(i_pts[k,:] - j_pts[l,:]) < 1:
                                rowindices = np.append(rowindices, l)
                                rowdata = np.append(rowdata, Int_obj.linfunc.gram_functions[i][j](i_pts[k,:],
                                                                                                  j_pts[l,:]))
                            if True:
                                np.save(blockpath + 'row' + str(k) + 'data', rowdata)
                                np.save(blockpath + 'row' + str(k) + 'indices', rowindices)
                lastrow=0

    return build_sparse_matrix(Int_obj, filepath)

###########################################################################################################

def build_sparse_matrix(Int_obj, filepath):
    """
    Returns the sparse Gram matrix after all rows/indices have been calculated
    :param Int_obj: Interpolation object
    :param filepath: filepath to retrieve data
    :return: sparse Gram matrix
    """
    print('Building sparse matrix...')

    len_temp = len(Int_obj.linfunc.gram_functions) # Number of row/column blocks in Gram matrix

    for i in range(len_temp):
        for j in range(len_temp):
            blockpath = filepath + '/Block' + str(i) + '-' + str(j) + '/'
            if j==i:    # Build coo matrix
                rows = np.array([],dtype=np.uint16)
                cols = np.array([],dtype=np.uint16)
                data = np.array([])
                num_points_in_dataobj = Int_obj.data_points[i].numpoints
                for k in range(num_points_in_dataobj):
                    readrowdata = np.load(blockpath + 'row' + str(k) + 'data.npy')
                    readrowindices = np.load(blockpath + 'row' + str(k) + 'indices.npy')
                    cols = np.append(cols, readrowindices)
                    rows = np.append(rows, k*np.ones((len(readrowindices),),dtype=np.uint16))
                    data = np.append(data, readrowdata)
                    assert readrowindices[0]==k # First element should be (nonzero) diagonal element
                    rows = np.append(rows, readrowindices[1:])    # Don't include diagonal element
                    cols = np.append(cols, k*np.ones((len(readrowindices)-1,),dtype=np.uint16))
                    data = np.append(data, readrowdata[1:])

                save_sparse_csr(filepath + '/Block' + str(i) + '-' + str(i), sparse.coo_matrix((data, (rows,cols)),
                                                   shape=(num_points_in_dataobj,num_points_in_dataobj)).tocsr())

            elif j > i:
                rowindices = np.array([], dtype=np.uint16)
                rowindptr = np.array([0], dtype=np.uint16)
                data = np.array([])
                num_points_in_dataobj_i = Int_obj.data_points[i].numpoints  #Number of rows in block
                num_points_in_dataobj_j = Int_obj.data_points[j].numpoints  #Number of cols in block
                for k in range(num_points_in_dataobj_i):
                    readrowdata = np.load(blockpath + 'row' + str(k) + 'data.npy')
                    readrowindices = np.load(blockpath + 'row' + str(k) + 'indices.npy')

                    rowindptr = np.append(rowindptr, np.array([rowindptr[-1] + len(readrowindices)],dtype=np.uint16))
                    rowindices = np.append(rowindices, readrowindices)
                    data = np.append(data, readrowdata)

                save_sparse_csr(filepath + '/Block' + str(i) + '-' + str(j),
                                sparse.csr_matrix((data, rowindices, rowindptr),
                                                  shape=(num_points_in_dataobj_i, num_points_in_dataobj_j)))
                save_sparse_csr(filepath + '/Block' + str(j) + '-' + str(i),
                                sparse.csr_matrix((data, rowindices, rowindptr),
                                                  shape=(num_points_in_dataobj_i, num_points_in_dataobj_j)).T)

        return sparse.vstack([sparse.hstack([load_sparse_csr(filepath + '/Block' + str(i) + '-' + str(j) + '.npz')
                                             for j in range(len_temp)]) for i in range(len_temp)])

def save_sparse_csr(filename,array):
    np.savez(filename, data = array.data , indices=array.indices, indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

def save_sparse_coo(filename,array):
    np.savez(filename, data = array.data , row=array.row, col =array.col, shape=array.shape )

def load_sparse_coo(filename):
    loader = np.load(filename)
    return sparse.coo_matrix((loader['data'], (loader['row'], loader['col'])), shape = loader['shape'])

##############################################################################################################

def gram_Wendland(Int_obj, save_rows=[False, None]):
    """
    Function uses compact support of Wendland function to create the Gram matrix from the
    Int_obj.linfunc and Int_obj.data_points attributes
    Assumes that the Int_obj.gram_functions are zero for
    Int_obj.K.c * np.linalg.norm(i_pts[k,:] - j_pts[l,:]) >= 1
    No vectorisation
    :param Int_obj: Interpolation object
    :param save_rows: Option to save Gram matrix block-by-block, row-by-row. 2nd list element is path
    :return: Gram matrix
    """
    # Use compact support to speed up Gram matrix population
    len_temp = len(Int_obj.linfunc.gram_functions) # Number of row/column blocks in Gram matrix

    if save_rows[0]:
        gramfile_path = save_rows[1] + '/Gram matrix'

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
                    if save_rows[0]==True:
                        gramfileblock_path = gramfile_path + '/Block' + str(i) + '-' + str(j)
                        if not os.path.exists(gramfileblock_path): os.makedirs(gramfileblock_path)
                        np.save(gramfileblock_path, list_of_rows[j][i].T)
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
                        if save_rows[0]==True:
                            gramfileblockrow_path = gramfile_path + '/Block' + str(i) + '-' + str(j) + '/'
                            if not os.path.exists(gramfileblockrow_path): os.makedirs(gramfileblockrow_path)
                            np.save(gramfileblockrow_path + 'row' + str(k), off_diag_block[k,:])
                    row.append(off_diag_block)
                    if save_rows[0]==True:
                        gramfileblock_path = gramfile_path + '/Block' + str(i) + '-' + str(j)
                        if not os.path.exists(gramfileblock_path): os.makedirs(gramfileblock_path)
                        np.save(gramfileblock_path, off_diag_block)

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
            if save_rows[0]==True:
                gramfileblockrow_path = gramfile_path + '/Block' + str(i) + '-' + str(i) + '/'
                if not os.path.exists(gramfileblockrow_path): os.makedirs(gramfileblockrow_path)
                np.save(gramfileblockrow_path +'row' +str(j), diag_block[j,:])
        list_of_rows[i][i] = diag_block
        if save_rows[0]==True:
            gramfileblock_path = gramfile_path + '/Block' + str(i) + '-' + str(i)
            if not os.path.exists(gramfileblock_path): os.makedirs(gramfileblock_path)
            np.save(gramfileblock_path, list_of_rows[i][i])
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
