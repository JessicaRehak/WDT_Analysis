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
        abs_location = os.path.abspath(os.path.expanduser(location))
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

    def get_avg(self, label, grp_entry, n=0):
        """ Returns the average FOM from the last `n` values of the
        Serpent parameter provided.
        
        :param label: Serpent 2 output parameter
        :type label: string

        :param grp_entry: the energy group of interest for the average. \
        Does not support multiple groups or matrix entry format, i.e. (1,1).        
        :type grp_entry: int

        :param n: the number of data points to be used to calculate the average
        :type n: int

        """
        data = self.__val_vs__(label, grp_entry, True, True)
        
        # Sort by cycle number
        data = data[data[:,0].argsort()]
        
        return np.mean(data[-n:,1])



    def get_collapse(self, label, grps, fom = True, cycle = True):
        """ Returns a combined FOM or error value for all the groups requested.

        :param label: Serpent 2 output parameter
        :type label: string

        :param grps: The groups to be combined.
        :type grps: list(int)

        :param fom: if True (default) returns the FOM, otherwise returns \
        error.
        :type fom: bool

        :param cycle: if True (default) returns cycle number in the first column, \
        otherwise returns cpu time.
        :type cycle: bool

        :returns: an array with cycles/cpu in the first column, and error \
        or FOM for the combined groups in the second column.
        :rtype: :class:`numpy.ndarray`

        """
        sum = 0
        for grp in grps:
            val = self.__val_vs__(label, grp, cycle, fom = False)
            if fom:
                sum += np.power(val[:,1:2],2)
            else:
                sum += val[:,1:2]

        sum = np.hstack((val[:,0:1], sum))

        if fom:
            # Get CPU
            cpu = self.__val_vs__(label, grp, cycle = False, fom = False)[:,0:1]
            zeros = (sum[:,1:2] == 0)
            sum[:,1:2] = np.power(np.multiply(sum[:,1:2], cpu),-1)
            sum[:,1:2][zeros] = 0
            
        return sum

    def get_collapse_avg(self, label, grps, n = 0):
        """ Returns the average FOM from the last _n_ values of the Serpent \
        parameter provided for multiple groups. This operates like :meth:`analysis.fom.Analyzer.get_avg` \
        but first combines the FOM for the groups and then calculates the average.

        :param label: Serpent 2 output parameter
        :type label: string

        :param grp: the energy groups of interest for the combined average
        :type grp_entry: int

        :param n: the number of data points to be used to calculate the average
        :type n: int
        
        """
        data = self.get_collapse(label, grps, True, True)

        data = data[data[:,0].argsort()]
        
        return np.mean(data[-n:,1])


    def get_data(self, label, grp_entry, fom = True, plot = False, cycle = True):
        """ Returns the an array with the error and cycle number for
        analysis of error for a given Serpent 2 output parameter
        and group number (if the parameter has multiple groups).

        :param label: Serpent 2 output parameter
        :type label: string

        :param grp_entry: The energy group(s) of interest or the matrix \
                          entries of interest.
        :type grp: int or list(int) if group, tuple(int, int) or list(tuple(int,int)) if entries

        :param fom: If True, returns Figure of merit, otherwise returns error.
        
        :param plot: If True, plots.
        :type plot: bool

        :param cycle: If True (default), returns the cycle number in \
                     the first column. Otherwise, returns the cpu time.
        :type cycle: bool

        :returns: :class:`numpy.array` with the cycles/cpu in the first column \
                  and error in the second column. If multiple groups are passed \
                  They will be horizontally concatenated in this manner (ex: second \
                  group specified will have data in the third column) """
        
        if (type(grp_entry) is list and type(grp_entry[0]) is int) or type(grp_entry) is int:
            if fom:
                data = self.__val_vs__(label, grp_entry, cycle, fom = True)
            else:
                data = self.__val_vs__(label, grp_entry, cycle, fom = False)
            group = True
        else:
            if fom:
                data = self.__mat_vs__(label, grp_entry, cycle, fom = True)
            else:
                data = self.__mat_vs__(label, grp_entry, cycle, fom = False)
            group = False

        if plot:
            if cycle:
                xlabel = 'Cycles'
            else:
                xlabel = 'CPU time'

            if fom:
                ylabel = 'FOM for ' + label.replace('_',' ')
                title = ylabel + ' vs. ' + xlabel
            else:
                ylabel = 'Error in ' + label.replace('_',' ')
                title = ylabel + ' vs. ' + xlabel

            if self.name != "":
                title = title + " (" + self.name + ")"
            
            if group:
                labels = self.__grp_label__(grp_entry)
            else:
                labels = self.__entry_label__(grp_entry)
            
            ax = self.__plot_me__(data, xlabel, ylabel, title, labels)
            ax.set_xscale('log')                            

            if not fom:
                ax.set_yscale('log')
        return data                

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
        """ Returns a list of the filenames for all files uploaded by the analyzer

        :rtype: list(string)
        """
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
        plt.grid(True,which='both',color='0.5')
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
    """ An object that contains two :class:`analysis.fom.Analyzer` objects
    and generates plots and comparisons for the results from each. 

    :param dirs: list of strings with the location of the data sets.
    :type dirs: list(string)

    :param names: list of strings that are the chosen names for the data sets
    :type names: list(string)

    :param verb: When True, shows all filenames as they are uploaded, \
                 useful to ensure initialization doesn't hang.
    :type verb: bool
    """
    
    def __init__(self, dirs, names, verb = False):
        assert len(dirs) == len(names), "Number of directories and names must match"
        self.data = [Analyzer(dir, names[i], verb) for i, dir in enumerate(dirs)]

    def add(self,dir,name, verb = False):
        self.data.append(Analyzer(dir,name,verb))
        
    def ratio(self, label, grp, n_pts):
        """ Returns an array with the ratio of the average FOM for the
        parameter in label compared to the first analyzer in the data
        object.
        """

        names = [d.name for d in self.data]
        data = [d.get_avg(label,grp,n_pts) for d in self.data]
        
        if data[0] != 0:
            data /= data[0]

        return names,data

    def collapse_ratio(self, label, grps, n_pts):
        """ Returns an array with the ratio of the average FOM for the
        parameter in label compared to the first analyzer in the data
        object. Collapses groups grps.
        """

        names = [d.name for d in self.data]
        data = [d.get_collapse_avg(label,grps,n_pts) for d in self.data]
        
        if data[0] != 0:
            data /= data[0]

        return names,data
        

    def compare(self, labels=[], mat_labels=[], n_pts=100):
        """ For each Serpent 2 output parameter specified, this will
        average the FOM using the last n_pts for both data sets to get
        an average FOM, and return a report indicating which data set
        had the higher FOM for each parameter.

        :param label: Serpent 2 output parameter
        :type label: list(string) or string

        :param mat_labels: Serpent 2 output parameters that are matrices
        :type label: list(string) or string

        :param n_pts: The number of points that should be used when comparing \
        the data sets.
        :type n_pts: int

        """

        if type(labels) is not list:
            labels = [labels]

        if type(mat_labels) is not list:
            mat_labels =[mat_labels]

        # Get the number of groups from 'INF_FLX'
        n = np.shape(self.data[0].data[0].get_data('INF_FLX'))[1]
        names = [d.name for d in self.data]
            
        for label in labels:
            for i in range(1,n+1):
                # Get data
                data = [d.get_avg(label,i,n_pts) for d in self.data]
                sort = np.sort(np.array(data))[::-1]
                if sort[1] > 1e-16:
                    r = sort[0]/sort[1]
                    zero = 0
                    ratio = str(r)
                else:
                    r = sort[0]
                    zero = 1
                    ratio = str(r) + " (Divide by zero)"

                if r > 1.1 or zero == 1:
                    print label + " Group: " + str(i) + ": " + names[np.argmax(data)] + " Ratio: " + ratio

        for label in mat_labels:
            for i in range(1,n+1):
                for j in range(1,n+1):
                    data = [d.get_avg(label,i+j-1,n_pts) for d in self.data]
                    sort = np.sort(np.array(data))[::-1]
                    if sort[1] > 1e-16:
                        r = sort[0]/sort[1]
                        zero = 0
                        ratio = str(r)
                    else:
                        r = sort[0]
                        zero = 1
                        ratio = str(r) + " (Divide by zero)"
                    if r > 1.1 or zero == 1:
                        print label + " Entry: (" + str(i) + "," + str(j) + "): " + names[np.argmax(data)] + " Ratio: " + ratio
        

    def plot(self, label, grp_entry, cycle = True, fom = True, show_avg=False, avg_n=100):
        """ Plots the given Serpent 2 output parameter for the specified groups
        for both sets of data.

        :param label: Serpent 2 output parameter
        :type label: string

        :param grp_entry: The energy group(s) of interest or the entries \
                          in the matrix of interest.
        :type grp_entry: groups: int or list(int), entries: tuple(int, int) or list(tuple(int,int))

        :param cycle: If True (default) plots against cycle number, otherwise CPU time.
        :type cycle: bool

        :param fom: If True (default) plots log-log plot of FOM, otherwise \
                    lin-log plot of error.
        :type fom: bool

        :param show_avg: If True, plot the average, as calculated using the final \
                         avg_n points of the dataset.
        :type show_avg: bool

        param avg_n: The number of points used to calculate the average
        :type avg_n: int
        
        """

        if (type(grp_entry) is list and type(grp_entry[0]) is int) or type(grp_entry) is int:
            group = True
        else:
            group = False
            
        data_sets = [d.get_data(label, grp_entry, fom=fom, plot=False,
                                cycle=cycle) for d in self.data]
        if cycle:
            xlabel = 'Cycles'
        else:
            xlabel = 'CPU time'

        if fom:
            ylabel = 'FOM for ' + label
            title = ylabel + ' vs. ' + xlabel
        else:
            ylabel = 'Error in ' + label
            title = ylabel + ' vs. ' + xlabel

        if group:
            labels = self.__grp_label__(grp_entry)
        else:
            labels = self.__entry_label__(grp_entry)
            
        ax = self.__multi_plot__(data_sets, xlabel, ylabel, title, labels)

        ax.set_xscale('log')

        if not fom:
            ax.set_yscale('log')
            

    def __multi_plot__(self, data_sets, xlabel, ylabel, title, labels):
        colors = self.__plot_setup__(xlabel, ylabel, title)
        
        for i in range(0,len(data_sets)):
            to_plot = data_sets[i]
            if i == 0:
                plot_color = colors[1]
                
            for j in range(1,np.shape(to_plot)[1]):
                plt.plot(to_plot[:,0], to_plot[:,j], '.', color =
                         plot_color[i*j + j - 1], label=labels[j-1] +
                         " (" + self.data[i].name + ")")
                
            plot_color = colors[0]
        plt.legend(loc = 'best', markerscale = 2.0)
        
        return plt.gca()

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
        gray = [(96, 99, 106), (165, 172, 175), (65, 68, 81), (143, 135, 130),
                (207, 207, 207)]
        for i in range(len(tableau)):  
            r, g, b = tableau[i]  
            tableau[i] = (r / 255., g / 255., b / 255.)
        for i in range(len(gray)):  
            r, g, b = gray[i]  
            gray[i] = (r / 255., g / 255., b / 255.)  
        return [tableau, gray]
        
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
