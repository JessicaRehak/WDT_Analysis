"""
.. module:: core
    :synopsis: The core analysis tools

.. moduleauthor:: Joshua Rehak <jsrehak@berkeley.edu>

"""

from pyne import serpent 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, sys
import warnings as warnings
import math as math
import pandas as pd
from IPython.display import display, HTML

class DataFile():
    """An object containing the data from a Serpent 2 output file
    (`_res.m`). When created, it will seek the provided filename and
    ingest all the data, stored in dictionary format.

    :param file_name: filename to be ingested.
    :type file_name: string
    """
    
    def __init__(self,file_name):
        assert os.path.exists(file_name), "File does not exist"
        self.data = serpent.parse_res(file_name)
        self.filename = file_name
        self.cpu = self.data['TOT_CPU_TIME'][0]
        
    def get_cpu(self):
        """Returns a float with the total CPU time"""
        return self.cpu
    
    def get_filename(self):
        """Returns a string with the filename"""
        return self.filename
    
    def all_data(self):
        """Returns the dictionary with all the `res.m` data"""
        return self.data
    
    def get_data(self, label, err = False, reshape = False):
        """Returns an array with the specified output data,
        either the values themselves or their associated error.
        
        :param label: the desired Serpent output parameter.
        :type label: string

        :param err: If False (default), returns the actual values of \
                    the specified parameter, otherwise returns the \
                    error associated.
        :type err: bool, optional

        :param reshape: If True, attempts to reshape the specified data \
                        into a matrix, such as for scattering matrices.
        :type reshape: bool, optional

        :returns: :any:`numpy.array` of dimension two.

        """
        
        # Returns the data contained in the res_m field labeled with label
        try:
            #data = self.__get_val__(self.data[label],err)
            data = self.__get_val__(label,err)
            if reshape:
                return self.__reshape__(data)
            else:
                return data
        except KeyError:
            raise KeyError('Invalid serpent2 res_m label')
            
    def get_fom(self, label, reshape=False):
        """Returns an array with the FOM for the the specified output
        parameter. Total CPU time :math:`T` in minutes, and the error
        :math:`\sigma` is read directly from the file. The FOM is
        calculated using:

        .. math::

           FOM = 1/(\sigma^2 T)

        If there is no error with the associated output parameter (such as
        Total CPU time) or if the error associated with a parameter is
        0, returns 0.
        
        :param label: the desired Serpent output parameter.
        :type label: string

        :param reshape: if False (default), returns an array, otherwise \
                        attempts to reshape into a square matrix.
        :type reshape: bool
        
        """
        
        try:
            data = []
            #errors = self.__get_val__(self.data[label],err = True)
            errors = self.__get_val__(label,err = True)
            for error in errors[0]:
                if error != 0:
                    data.append(np.power(self.cpu*np.power(error,2), -1))
                else:
                    data.append(0)
            data = np.array(data, ndmin=2)
            if reshape:
                return self.__reshape__(data)
            else:
                return data
        except KeyError:
            raise KeyError('Invalid serpent2 res_m label')
    
    def __get_val__(self,label, err = False):
        array = self.data[label]
        shape = np.shape(array)
        if shape == (1,1) or len(shape) == 1:
            if err:
                return np.array([0],ndmin=2)
            return np.array(array,ndmin=2)
        else:
            return np.array([array[0,i] for i in range(shape[1]) if i % 2 == err],ndmin=2)
        
    def __reshape__(self,array):
        # Reshapes into a square matrix
        shape = np.shape(array)
        if shape == (1,):
            warnings.warn('This appears to be a single value, not reshaping')
            return array
        else:
            n = np.sqrt(shape[1])
            if n.is_integer():
                n = int(n)
                return np.reshape(array,(n,n))
            else:
                warnings.warn('This does not appear to be a square matrix, skipping reshape')
                return array

