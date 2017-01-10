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

class DataSet():
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

class ParamData:
    """ An object containing all the :class:`analysis.core.DataSet` objects for a particular
    set of surface tracking and weighted-delta tracking threshold values.
    When created, it will find all the `_res.m` files in the appropriate
    directory.

    :param st_th: the desired surface tracking threshold
    :type st_th: float
    :param wdt_th: the desired weighted delta-tracking threshold
    :type wdt_th: float
    :param base_dir: the location of the `wdt_runs` repo
    :type base_dir: string
    
    """
    def __init__(self, st_th, wdt_th, base_dir):
        # Generate directory name and verify existence
        dir_name = base_dir + "wdt_runs/S0" + str("%03.f" % (st_th*1000)) + "/W" + str("%04.f" % (wdt_th*1000)) + "/runs/"
        assert os.path.exists(dir_name), "Error: directory for those parameters does not exist"

        # Generate filenames and verify existence, skip if they don't
        self.dataSets = []

        # Upload data, generate warnings if files do not exist
        for i in range(1,11):
            file_name = dir_name + 'run' + str(i) + '_res.m'
            if os.path.exists(file_name):
                self.dataSets.append(DataSet(file_name))
            else:
                warnings.warn(file_name + ' does not exist, skipping')
                
        # Get the number of groups from the infinite flux tally
        self.n = np.shape(self.dataSets[0].get_data('INF_FLX'))[1]
        self.st_th = st_th
        self.wdt_th = wdt_th
        
    def cpu(self):
        """Returns an array with the mean and standard deviation of total CPU
        time for the ten simulations. The first element of the array is
        the mean, and second is the standard deviation.

        :returns: :class:`numpy.array`

        """
        cpu = np.zeros([2,1])
        cpu[0,0] = np.mean([data.get_cpu() for data in self.dataSets])
        cpu[1,0] = np.std([data.get_cpu() for data in self.dataSets])
        return cpu

    def value_stat(self, label, err = False, plot = False):
        """For the specified parameter, this returns numpy array in which the
        first row contains the mean value (or mean error) from the ten
        simulations for each energy group. The second row contains the
        standard deviation for the values (or errors).

        :param label: specifies the Serpent 2 output parameter.
        :type label: string

        :param err: if False (default), returns the mean values, otherwise \
                    returns the mean error.
        :type err: bool
        
        :param plot: If True, will plot the FOM by energy group.
        :type plot: bool

        """
        value = np.zeros([2,self.n])

        for i in range(self.n):
            group_values = self.__group_vals__(label,i, err = err)
            value[0,i] = np.mean(group_values)
            value[1,i] = np.std(group_values)
        if plot:
            plt.errorbar(range(1,self.n+1),value[0,:],fmt='.b',yerr=value[1,:])
            plt.ylabel(label)
            plt.xlabel('Group')
        return value
    
    def fom_stat(self, label, plot=False):
        """ For the specified parameter, this returns numpy array in which
        the first row contains the mean FOM from the ten simulations for
        each energy group. The second row contains the standard deviation
        for those mean FOM values.

        :param label: specifies the Serpent 2 output parameter.
        :type label: string
        :param plot: If True, will plot the FOM by energy group.
        :type plot: bool

        """
        fom = np.zeros([2,self.n]) 
        # Vector quantity
        for i in range(self.n):
            group_fom = self.__group_fom__(label,i)
            fom[0,i] = np.mean(group_fom)
            fom[1,i] = np.std(group_fom)
        if plot:
            plt.errorbar(range(1,self.n+1),fom[0,:],fmt='.b',yerr=fom[1,:])
            plt.yscale('log')
            plt.ylabel('FOM')
            plt.xlabel('Group')
        return fom
    
    def mat_stat(self, label, mean=True, plot = False):
        """For the specified matrix parameter (such as scattering matrices),
        this returns a numpy array of the same shape of the parameter
        matrix. Each entry in this matrix is the mean (or standard deviation)
        FOM for that entry of the parameter's matrix.

        .. note: The mean is calculated only using non-zero FOM values \
                 as a 0 FOM indicates that not enough statistics were \
                 collected to produce an error.

        :param label: specifies the Serpent 2 output parameter.
        :type label: string

        :param mean: if True (default) returns the mean FOM, otherwise returns \
                     the standard deviation of the FOM.
        :type mean: bool

        :param plot: if True, plots the FOM matrix using :any:`matplotlib.pyplot.matshow`
        :type plot: bool
        """
        fom = np.zeros([self.n,self.n])

        #list of numpy arrays
        all_data = [data.get_fom(label, reshape = True) for data in self.dataSets]

        for i in range(self.n):
            for j in range(self.n):
                if mean:
                    fom[i,j] = self.__nz_mean__(np.array([d[i,j] for d in all_data]))
                else:
                    fom[i,j] = np.std(np.array([d[i,j] for d in all_data]))
        if plot:
            logdata = np.where(fom > 0, np.log(fom), np.zeros(np.shape(fom)))
            im = plt.imshow(logdata, interpolation='none')
            plt.title(label + " for ST: " + str(self.st_th) + " WDT: "
                      + str(self.wdt_th), size=10)
            plt.colorbar(im)
        else:
            return fom
                                                
    def __nz_mean__(self, a):
        """ Returns the mean of the non-zero elements """
        nz = np.count_nonzero(a)
        if nz == 0:
            return 0
        else:
            return np.sum(a)/nz
        
        
    def __group_vals__(self, label, grp, err = False, isMatrix = False):
         """ Returns the values of label for each run for that group """
         return np.array([data.get_data(label, reshape = isMatrix, err = err)[:,grp] for data in self.dataSets])
    
    def __group_fom__(self, label, grp):
        return np.array([data.get_fom(label)[:,grp] for data in self.dataSets])

class Analyzer:
    """This class is used to analyze and compare data across multiple surface tracking
    and WDT tracking values.

    :param st_vals: The surface tracking values to be analyzed
    :type  st_vals: list

    :param wdt_vals: The weighted delta tracking values to be analyzed
    :type  wdt_vals: list

    :param base_dir: The base directory where the `wdt_runs` repo is
    :type  base_dir: string

    :param base_st: The base value of the surface tracking threshold \
                    used when calculating relative improvement, defaults to the
                    Serpent default of 0.1.
    :type base_st: float

    :param base_wdt: The base value of the weighted delta-tracking threshold \
                     used when calculating relative improvement, defaults to 0.1.
    :type base_wdt: float
    
    .. note:: If `base_st` and `base_wdt` are not specified, the base \
              case will default to no weighted delta-tracking.

    """
    
    def __init__(self, st_vals, wdt_vals, base_dir, base_st=0.1, base_wdt=0.1):
        """Initializes the analyzer, this will automatically ingest the
        Serpent output files to be analyzed.

        

        """
        # Creates an analyzer to compare the parameters in st_vals and wdt_vals to a given
        # base case. The base case is assumed to be no WDT but it can be changed
        self.dataSets = []
        
        # Get baseline data, 0 will always be baseline
        self.dataSets.append(ParamData(base_st,base_wdt,base_dir))
        
        for st_th in st_vals:
            for wdt_th in wdt_vals:
                if not ((wdt_th == base_wdt) and (st_th == base_st)):
                    self.dataSets.append(ParamData(st_th, wdt_th, base_dir))
        
        self.st_vals = st_vals;
        self.wdt_vals = wdt_vals;
        self.n = self.dataSets[0].n
    
    def histogram(self,labels,mean=True):
        """ Compares a list of serpent output parameters across all the data sets stored
        in the analyzer. For each parameter specified, the top five ST/WDT
        combinations are identified. Returns a histogram showing the ST/WDT
        combinations that were in the top five the most times.

        :param labels: a list of serpent output parameters to be compared
        :type labels: list of strings

        :param mean: if `True`, compares the mean FOM for each output parameter. Otherwise \
                     compares the standard deviation.
        :type mean: bool

        :returns: :class:`pandas.DataFrame`
        
        """
        
        histogram = pd.DataFrame()
        for label in labels:
            if mean:
                df = self.data_frame(label,mean=True,style=False,rel=False).sort_values(by='Ratio', ascending = False)[:5]
            else:
                df = self.data_frame(label,mean=False,style=False, rel=False).sort_values(by='Ratio', ascending = True)[:5]
            histogram = histogram.append(df)
        return histogram[['ST Thr','WDT Thr','Sum']].groupby(['ST Thr','WDT Thr']).count()
        
        
    def data_frame(self,label,rel=True,mean=True,style=True):
        """Returns a DataFrame with the selected serpent output parameter
        for all groups.

        :param label: the desired Serpent output parameter.
        :type label: string

        :param rel: returns relative improvement if `True` or absolute values \
                    if `False`.
        :type rel: bool

        :param mean: default True, If mean should be returned or standard deviation
        :type mean: bool

        :param style: returns an html :class:`pandas.formats.style.Styler` object if `True` with color \
                      coding. If `False`, returns a :class:`pandas.DataFrame`.
        :type style: bool

        """
        
        def color_grad(val):
            if val < 0:
                color = '#E5000A'
            elif val == 0:
                color = 'black'
            elif val <= 0.25:
                color = '#00CE4F'
            elif val <= 1.0:
                color = '#009FC6'
            elif val <= 2.0:
                color = '#0000BF'
            else:
                color = '#DDA800'
            return 'color: %s' % color
        
        def rev_grad(val):
            return color_grad(-val)
        
        def color_neg(val):
            if val < 0:
                color = '#E5000A'
            else:
                color = 'black'
            return 'color: %s' % color
        
        def color_pos(val):
            return color_neg(-val)
        
        cpu_time = label == 'TOT_CPU_TIME'
        FOM_DF = pd.DataFrame()
        FOM_DF["ST Thr"] = [data.st_th for data in self.dataSets]
        FOM_DF["WDT Thr"] = [data.wdt_th for data in self.dataSets]
        
        if cpu_time:
            FOM_DF['CPU Time'] = self.__rel_cpu__()
        else:
            if rel:
                for i in range(0,self.n):
                    if mean:
                        FOM_DF[str(i+1) + " m"] = self.__rel_mean__(label,i)
                    else:
                        FOM_DF[str(i+1) + " st"] = self.__rel_stdv__(label,i)
            else:
                for i in range(0,self.n):
                    if mean:
                        FOM_DF[str(i+1) + " m"] = self.__get_mean__(label,i)
                    else:
                        FOM_DF[str(i+1) + " st"] = self.__get_stdv__(label,i)

        # Generate Column list, remove first two columns
        col_list = list(FOM_DF)[2:]
        
        FOM_DF['Sum'] = FOM_DF[col_list].sum(axis=1)
        if not rel:
            FOM_DF['Ratio'] = FOM_DF['Sum']/(FOM_DF['Sum'][0])
        FOM_DF = FOM_DF.fillna(0)
        
        if style:
        #print FOM_DF.to_string(index=False)
            s = FOM_DF.style.\
                format({'ST Thr': '{:.3f}','WDT Thr': '{:.2f}'})
                
            if mean and not cpu_time:
                s.applymap(color_grad,subset=col_list)
                s.applymap(color_neg, subset='Sum')
            else:
                s.applymap(rev_grad,subset=col_list)
                s.applymap(color_pos, subset='Sum')
                
            for col in col_list:
                s.format({col:'{:.4f}'})
                
            s.format({'Sum': '{:.4f}'})
                

            return s
        else:
            return FOM_DF
        
    def plot_setup(self,xlabel,ylabel):
            
        self.base_color = ([0.0,107.0/255,164.0/255])
        self.other_color = ([1.0,128.0/255,14.0/255])
        plt.figure(figsize=(12, 9))
        ax = plt.subplot(111)  
        ax.spines["top"].set_visible(False)  
        ax.spines["right"].set_visible(False)
        ax.get_xaxis().tick_bottom()  
        ax.get_yaxis().tick_left()
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylabel(ylabel,fontsize=16)
        plt.xlabel(xlabel,fontsize=16)
        
    def plot_cpu(self,st_th):
        """ Plots CPU time against weighted delta-tracking threshold for a given
        surface tracking threshold, this will be shown relative to the base case
        specified when the analyzer was created.

        :param st_th: the surface tracking threshold of interest.
        :type st_th: float
       
        """

        self.plot_setup('WDT Threshold','Relative Difference in Total CPU Time')
        
        cpu_df = self.data_frame('TOT_CPU_TIME',style=False)
        data = cpu_df.loc[cpu_df['ST Thr'] == st_th]

        plt.plot(data['WDT Thr'], data['Sum'],'o',color=self.base_color, ms=6, mec=self.base_color)
        plt.axhline(color = 'black', ls='dashed')
        plt.title('Relative difference in total CPU time by WDT threshold for surface tracking threshold ' + str(st_th),fontsize=14)
        plt.xlim([0,1.1])
        return plt
        
    def plot(self,label, st_th, wdt_th, st_base = 0.1, wdt_base = 0.1):
        """ Plots a given Serpent parameter by energy group, for a given
        combination of ST/WDT threshold. The plot uses the mean
        values for the points and standard deviations for error bars. Also
        plots a base case, defaulting to no WDT.

        :param label: the desired Serpent output parameter
        :type label: string

        :param st_th: the surface tracking threshold to plot.
        :type st_ths: float

        :param wdt_th: the weighted delta-tracking threshold to plot.
        :type wdt_th: float

        :param st_base: default 0.1, surface tracking threshold for the base case.
        :type st_base: float

        :param wdt_base: default 0.1, weighted delta-tracking threshold for the base case.
        :type wdt_base: float

        """

        
        # Ensure that the axis ticks only show up on the bottom and left of the plot.  
        # Ticks on the right and top of the plot are generally unnecessary chartjunk.  
        self.plot_setup('Neutron Group','Figure of Merit')
        
        for data in self.dataSets:
            if data.st_th == st_base and data.wdt_th == wdt_base:
                fom = data.fom_stat(label)
                plt.errorbar(range(1,self.n+1), fom[0,:],
                             yerr=fom[1,:],
                             label=str(data.st_th)+'/'+str(data.wdt_th),
                             fmt='^', ms=6,
                             capthick=1,mec=self.base_color,
                             color=self.base_color, capsize=5)
                
        for data in self.dataSets:
            if data.st_th == st_th and data.wdt_th == wdt_th:
                fom = data.fom_stat(label)
                plt.errorbar(range(1,self.n+1),fom[0,:],yerr=fom[1,:],
                             label=str(data.st_th)+'/'+str(data.wdt_th),
                             fmt='o',
                             ms=6,capthick=1,mec=self.other_color,
                             color=self.other_color, capsize=10)

        plt.yscale('log')

        #plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
        #  fancybox=True, shadow=True, ncol=5)
        plt.legend(loc='best', fontsize=14,numpoints=1)
        return plt

    def plot_mat(self, label, param_sets=[(0.1,0.1)], mean = True):
        """ Plot the FOM means or standard deviations for a matrix
        quantity (such as scattering matrices). A single figure with
        subplots for each param_set will be created.

        :param label: the desired Serpent output matrix parameter
        :type label: string

        :param param_sets: the desired st/wdt combinations that should \
                           be plotted. Each set should be in a tuple,
                           a maximum of 30 can be plotted.
        :type param_sets: list(tuple(float, float))

        :param mean: if True (default), will display the mean FOM, \
                     otherwise will display the standard deviation of \
                     the FOM.
        :type mean: bool

        """

        n = len(param_sets)

        assert n < 10, 'must pass less than 10 parameter sets'

        rows = int((n-1)/2) + 1

        plot_num = rows*100 + 20 + 1

        for data in self.dataSets:
            if (data.st_th, data.wdt_th) in param_sets:
                plt.subplot(plot_num)
                data.mat_stat(label, plot = True)
                plot_num = plot_num + 1
        plt.tight_layout()
        
    
    def __get_mean__(self,label, grp):
        return [data.fom_stat(label)[0][grp] for data in self.dataSets]
    
    def __get_stdv__(self,label,grp):
        return [data.fom_stat(label)[1][grp] for data in self.dataSets]

    def __rel_mean__(self,label,grp):
        # Returns the value RELATIVE to the base case in dataSets[0]
        vals = self.__get_mean__(label,grp)
        base_val = vals[0]
        return (vals-base_val)/base_val
    
    def __rel_cpu__(self):
        vals = np.array([data.cpu()[0,0] for data in self.dataSets])
        base_val = vals[0]
        return (vals-base_val)/base_val
    
    def __rel_stdv__(self,label,grp):
        vals = self.__get_stdv__(label,grp)
        base_val = vals[0]
        return (vals-base_val)/base_val
