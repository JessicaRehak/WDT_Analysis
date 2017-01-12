"""

.. module:: fom
     :synopsis: Tools for analyzing FOM convergence

.. moduleauthor:: Joshua Rehak <jsrehak@berkeley.edu>

"""

import numpy as np
import matplotlib.pyplot as plt
import os, sys
import math
import pandas as pd
import core

class Analyzer():
    """ An object containing multiple :class:`analysis.core.DataFile`
    objects with methods to analyze FOM convergence properties. All
    `res.m` files in a directory will be ingested when initialized,
    the intention is that each of these represents the same simulation
    at different cycle values.

    :param location: folder where the Serpent output files are located
    :type location: string

    :param name: desired name for this data set
    :type name: string, optional
    
    :param verb: if True, prints the name of the files uploaded
    :type verb: bool


    """

    def __init__(self, location, name = "", verb = False):
        self.name = name
        # Verify file location exists
        abs_location = os.path.abspath(location)
        assert os.path.exists(abs_location), "Folder does not exist"

        # Initialize data array
        self.data = []
        
        # Get all .m files
        for file_name in os.listdir(abs_location):
            if file_name[-2:] == '.m':
                if verb: print "Uploading: " + file_name
                file_loc = abs_location + '/' + file_name
                self.data.append(core.DataFile(file_loc))

        print "Uploaded " + str(len(self.data)) + " files."
                
    def err(self, label, grp = 1, plot = False, cycle = True):
        """ Returns the an array with the error and cycle number for
        analysis of error for a given Serpent 2 output parameter
        and group number (if the parameter has multiple groups).

        :param label: Serpent 2 output parameter
        :type label: string

        :param grp: The energy group of interest (default 1)
        :type grp: int or list(int)

        :param plot: If True, plots the error.
        :type plot: bool

        :param mode: If True (default), returns the cycle number in \
                     the first column. Otherwise, returns the cpu time.
        :type mode: bool

        :returns: :class:`numpy.array` with the cycles/cpu in the first column \
                  and error in the second column. If multiple groups are passed \
                  They will be horizontally concatenated in this manner (ex: second \
                  group specified will have data in the third column)
        """        
        err = self.__val_vs__(label, grp, cycle, fom = False)

        if plot:
            if cycle:
                xlabel = 'Cycles'
            else:
                xlabel = 'CPU Time'
                
            ax = self.__plot_me__(err, xlabel, 'ERR', 'ERR for ' +
                                  label + ' vs. ' + xlabel,
                                  labels = self.__grp_label__(grp))
            ax.set_xscale('log')
            ax.set_yscale('log')
        return err

    def err_mat(self, label, entry, plot = False, cycle = True):
        """ Returns the an array with the error and cycle number for
        analysis of error for a given Serpent 2 matrix output parameter for
        a given location (or locations) in the matrix.

        :param label: Serpent 2 output parameter
        :type label: string

        :param entry: The matrix entry of interest.
        :type entry: tuple(int, int) or list(tuple(int, int)))

        :param plot: If True, plots the error.
        :type plot: bool

        :param mode: If True (default), returns the cycle number in \
                     the first column. Otherwise, returns the cpu time.
        :type mode: bool

        :returns: :class:`numpy.array` with the cycles/cpu in the first column \
                  and error in the second column. If multiple matrix entries are passed \
                  They will be horizontally concatenated in this manner (ex: second \
                  matrix entry specified will have data in the third column)
        """        
        err_mat = self.__mat_vs__(label, entry, cycle, fom = False)

        if plot:
            if cycle:
                xlabel = 'Cycles'
            else:
                xlabel = 'CPU Time'
                
            ax = self.__plot_me__(err_mat, xlabel, 'ERR', 'ERR for ' +
                                  label + ' vs. ' + xlabel,
                                  labels = self.__entry_label__(entry))
            ax.set_xscale('log')
            ax.set_yscale('log')
        
        return err_mat

    def fom_mat(self, label, entry, plot = False, cycle = True):
        """ Returns the an array with the FOM and cycle number for
        analysis of error for a given Serpent 2 matrix output parameter for
        a given location (or locations) in the matrix.

        :param label: Serpent 2 output parameter
        :type label: string

        :param entry: The matrix entry of interest.
        :type entry: tuple(int, int) or list(tuple(int, int)))

        :param plot: If True, plots the error.
        :type plot: bool

        :param mode: If True (default), returns the cycle number in \
                     the first column. Otherwise, returns the cpu time.
        :type mode: bool

        :returns: :class:`numpy.array` with the cycles/cpu in the first column \
                  and error in the second column. If multiple matrix entries are passed \
                  They will be horizontally concatenated in this manner (ex: second \
                  matrix entry specified will have data in the third column)
        """        
        fom_mat = self.__mat_vs__(label, entry, cycle, fom = True)
        
        if plot:
            if cycle:
                xlabel = 'Cycles'
            else:
                xlabel = 'CPU Time'
                
            ax = self.__plot_me__(fom_mat, xlabel, 'FOM', 'FOM for ' +
                                  label + ' vs. ' + xlabel, labels =
                                  self.__entry_label__(entry))
            ax.set_xscale('log')

        return fom_mat
    
    def fom(self, label, grp = 1, plot = False, cycle = True):
        """ Returns the an array with the FOM and cycle number for
        analysis of convergence for a given Serpent 2 output parameter
        and group number (if the parameter has multiple groups).

        :param label: Serpent 2 output parameter
        :type label: string

        :param grp: The energy group of interest (default 1)
        :type grp: int or list(int)

        :param plot: If True, plots the FOM convergence.
        :type plot: bool

        :param mode: If True (default), returns the cycle number in \
                     the first column. Otherwise, returns the cpu time.
        :type mode: bool

        :returns: :class:`numpy.array` with the cycles/cpu in the first column \
                  and error in the second column. If multiple groups are passed \
                  They will be horizontally concatenated in this manner (ex: second \
                  group specified will have data in the third column)

        """
            
        fom = self.__val_vs__(label, grp, cycle, fom = True)

        if plot:
            if cycle:
                xlabel = 'Cycles'
            else:
                xlabel = 'CPU Time'
                
            ax = self.__plot_me__(fom, xlabel, 'FOM', 'FOM for ' +
                                  label + ' vs. ' + xlabel, labels =
                                  self.__grp_label__(grp))
            ax.set_xscale('log')
        return fom

    def __plot_me__(self, data, xlabel, ylabel, title, labels):
        colors = self.__plot_setup__(xlabel, ylabel, title)
        
        for i in range(1,np.shape(data)[1]):
            plt.plot(data[:,0], data[:,i], '.', color = colors[i-1], label=labels[i-1])
        plt.legend(loc = 'best')
        return plt.gca()
        

    def __val_vs__(self, label, grp = 1, cycle = True, fom = True):
        # Cast into a list if an integer is passed
        if type(grp) is not list:
            grp = [grp]
        
        ans = np.zeros((len(self.data), 1+len(grp)))
        for j, g in enumerate(grp):
            for i, d in enumerate(self.data):
            
                if fom:
                    ans[i,j+1] = d.get_fom(label)[0][g-1]
                else:
                    ans[i,j+1] = d.get_data(label, err = True)[0][g-1]
                
                if cycle:
                    ans[i,0] = d.get_data('CYCLE_IDX')[0][0]
                else:
                    ans[i,0] = d.get_cpu()

        return ans

    def __mat_vs__(self, label, entry, cycle = True, fom = True):

        if type(entry) is not list:
            entry = [entry]
        
        # Get size of the matrix from the first entry
        n = np.shape(self.data[0].get_data(label, err = True,
                                               reshape = True))[0]
        
        assert n != 1, "Reshape failed, invalid Serpent matrix parameter"

        loc = []
        for e in entry:
            assert e[0] <= n and e[1] <= n, "Invalid matrix location " + str(e)
            loc.append((e[0] - 1)*n + e[1])

        return self.__val_vs__(label, loc, cycle, fom)
        
        

    def __grp_label__(self, grp):
        
        if type(grp) is not list:
            return ["Group " + str(grp)]
        else:
            return ["Group " + str(g) for g in grp]

    def __entry_label__(self, entry):
        
        if type(entry) is not list:
            return ["Entry " + str(entry)]
        else:
            return ["Entry " + str(e) for e in entry]

                     
    def get_filenames(self):
        return [d.get_filename() for d in self.data]


    def __plot_setup__(self,xlabel,ylabel, title):
        self.base_color = ([0.0,107.0/255,164.0/255])
        self.other_color = ([1.0,128.0/255,14.0/255])
        plt.figure(figsize=(12, 9))
        ax = plt.subplot(111)  
        ax.spines["top"].set_visible(False)  
        ax.spines["right"].set_visible(False)
        ax.get_xaxis().tick_bottom()  
        ax.get_yaxis().tick_left()
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.ylabel(ylabel,fontsize=12)
        plt.xlabel(xlabel,fontsize=12)
        plt.title(title, y=1.08)
        tableau = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),  
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),  
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),  
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),  
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
        for i in range(len(tableau)):  
            r, g, b = tableau[i]  
            tableau[i] = (r / 255., g / 255., b / 255.)  
        return tableau

class Comparator:
    """An object that contains two :class:`analysis.fom.Analyzer` objects
    and generates plots comparing the results from each.

    :param dir: list of strings with the location of the data sets.
    :type dir: list(string)

    :param names: list of strings that are the chosen names for the data sets
    :type names: list(string)

    :param verb: When True, shows all filenames as they are uploaded, \
                 useful to ensure initialization doesn't hang.

    """
    def __init__(self, dirs, names, verb = False):
        assert len(dirs) == len(names), "Number of directories and names must match"
        self.data = [Analyzer(dir, names[i], verb) for i, dir in enumerate(dirs)]

    def plot_err(self, label, grp, cycle = True):
        """ Plots the given Serpent 2 output parameter for the specified groups
        for both sets of data

        :param label: Serpent 2 output parameter
        :type label: string

        :param grp: The energy group(s) of interest
        :type grp: int or list(int)

        :param cycle: If True (default) plots against cycle number, otherwise CPU time.
        :type cycle: bool
        
        """
        data_sets = [d.err(label, grp, False, cycle) for d in self.data]

        if cycle:
            xlabel = 'Cycles'
        else:
            xlabel = 'CPU time'
        
        self.__multi_plot__(data_sets, xlabel, 'Error in ' + label,
                            'Error in ' + label, self.__grp_label__(grp))

    def __muti_plot__(self, data_sets, xlabel, ylabel, title, labels):
        
        
        
    def __grp_label__(self, grp):
        
        if type(grp) is not list:
            return ["Group " + str(grp)]
        else:
            return ["Group " + str(g) for g in grp]
