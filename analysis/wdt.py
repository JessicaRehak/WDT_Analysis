"""

.. module:: wdt
     :synopsis: Tools for conducting FOM studies of the weighted delta-tracking
                method. Can be extended for any other parametric study conducted
                in Serpent

.. moduleauthor:: Joshua Rehak <jsrehak@berkeley.edu>

"""
import numpy as np
import os, sys
import core

class SerpentRun():
    """ An object containing multiple :class:`analysis.core.DataFile`
    generated by a single run of Serpent. All `res.m` files in a
    directory can be ingested, or only a portion.
    
    :param directory: location where the Serpent output files are
    :type location: string

    :param params: A list of the parameters for this run. Each must
                   be a tuple in the form (parameter name, value). This
                   is optional and is best used in a parametric study.
    :type params: list(tuple)

    :param cyc_cpu: Cycles/CPU for this parameter set
    :type cyc_cpu: float

    :param verb: if True, prints the name of each file uploaded
    :type verb: bool


    """
    def __init__(self, directory, params = [], cyc_cpu=1.0, verb=False):
        # Verify location exists
        abs_location = os.path.abspath(os.path.expanduser(directory))
        assert os.path.exists(abs_location), "Folder does not exist"

        # Initialize files array
        self.files = []
        
        # Ingest all .m files
        for filename in [x for x in os.listdir(abs_location) if x[-2:] == '.m']:
            if verb: print("Uploading: " + filename)
            self.files.append(core.DataFile(abs_location + '/' +
                                            filename))

        # Sort files based on cycle number
        self.files.sort()

        # Attributes
        ## Number of files
        self.n = len(self.files)
        assert self.n > 0, "No .m files in that location"

        ## Cycles
        self.cycles = np.array([file.cycles for file in self.files])

        ## CPU time
        self.cpus = np.array([file.cpu for file in self.files])
        
        ## Parameter list
        if type(params) is not list: params = [ params ]
        if len(params):
            assert all(isinstance(x, tuple) for x in params), \
                "Parameters must be tuples"
        self.params = params

        ## Original location
        self.loc = abs_location

        ## Cycles/CPU
        self.cyc_cpu = cyc_cpu
        
        print("Uploaded " + str(self.n) + " files.")

    def __str__(self):
        # Print information about the run
        ret_str = str(self.n) + ' .m files uploaded from: ' + self.loc\
                  + '\n' + 'Parameters:\n'
        for tup in self.params:
            ret_str += '\t' + str(tup[0]) + ':\t' + str(tup[1]) + '\n'
        return ret_str[:-1]

    def cyc_v_cpu(self):
        """ Returns an array with the differential cycles/cpu between
        each file. This array will be of a dimension one less than the
        number of files. """
        cyc = np.subtract(self.cycles[1:], self.cycles[:-1])
        cpu = np.subtract(self.cpus[1:], self.cpus[:-1])
        return np.divide(cyc, cpu)
        
       
    def fom(self, label, grp, cpu=True, cap=np.inf):
        """ Returns an array with the calculated FOM for a given Serpent 2
        output parameter and group number. Calculated using the CPU time
        or cycle number.

        :param label: Serpent 2 output parameter
        :type label: string

        :param grp: The energy group of interest. This is ENERGY GROUP not the entry
                    in the vector.
        :type grp: int

        :param cpu: If True (default), calculates the FOM using the cpu time.
                    Otherwise uses the cycle number.
        :type cpu: bool

        :param cap: Upper cap on the cycle number for data.
        :type cap: float

        """

        end = np.shape(self.cycles[self.cycles < cap])[0]
        
        return self.__fom__(self.get_error(label, grp),
                            (self.cpus if cpu else self.cycles))[:end]

    def fom_corr(self, label, grp, cap=np.inf):
        """ Returns an array with the calculated FOM for a given Serpent 2
        output parameter and group number. Calculated using the cycle number
        and then corrected using this SerpentRun's value for cyc_cpu

        """
        return self.cyc_cpu * self.fom(label, grp, cpu=False, cap=cap)

    def fom_std(self, *args, **kwargs):
        """ Calculates the standard deviation using half the data points (after
        application of cap if desired for the given label and group
        number

        :Keyword Arguments: same as for :func:`~analysis.wdt.SerpentRun.fom`
        
        """
        fom = self.fom(*args, **kwargs)
        return self.__std__(fom)

    def fom_std_corr(self, *args, **kwargs):
        """ Calculates the std deviation using half the data points of the
        _corrected_ FOM

        :Arguments: same as for :func:`~analysis.wdt.SerpentRun.fom_corr`
        """
        fom = self.fom_corr(*args, **kwargs)
        return self.__std__(fom)
    
    def get_error(self, label, grp):
        """ Returns an array with the value or error for a given
        Serpent 2 output parameter and group number

        :param label: Serpent 2 output parameter
        :type label: string

        :param grp: The energy group of interest. This is ENERGY GROUP not the entry
                    in the vector.
        :type grp: int

        """
        return np.array([file.get_data(label, err=True)[0][grp - 1]
                         for file in self.files])
    
    def __fom__(self, error, time):
        """ Calc cpu given error and time """
        return np.power(np.multiply(np.power(error,2), time),-1)

    def __std__(self, fom, start=0):
        if not start:
            start = int(np.floor(len(fom)/2.0))
        return np.std(fom[start:])
