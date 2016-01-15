import numpy as np
from scipy.spatial import distance
import scipy.sparse as sparse





def gram_Wendland(Int_obj):
        """
        Creates the Gram matrix from the self.linfunc and self.data_points attributes
        Checks if any of the Data objects are empty, if so removes them from the matrix
        and data_points and linfunc attributes
        Uses compact support of Wendland function, no vectorisation
        :param Int_obj: Interpolation object
        :return: Gram matrix
        """
        # Use compact support to speed up Gram matrix population
        len_temp = len(Int_obj.linfunc.gram_functions) # Number of rows/column blocks in Gram matrix

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
        Creates the Gram matrix from the self.linfunc and self.data_points attributes
        Checks if any of the Data objects are empty, if so removes them from the matrix
        and data_points and linfunc attributes
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
