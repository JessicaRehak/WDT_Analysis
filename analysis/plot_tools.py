import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import core
import fom

def fom_plot_setup(font_size=32, label_size=32):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    ax = plt.gca()
    plt.rc('font', size=font_size)
    ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
    #ax.spines["top"].set_visible(False)  
    #ax.spines["right"].set_visible(False)
    #ax.get_xaxis().tick_bottom()  
    #ax.get_yaxis().tick_left()
    plt.ylabel('Figure of merit', fontsize=label_size)
    plt.xlabel('$n$', fontsize=label_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    ax.tick_params(axis='y', labelsize=font_size)
    line = plt.gca().get_lines()[0]
    line.set_marker('.') 
    line.set_color('black')
    plt.grid(True,which='both',color='0.5')
    plt.xscale('linear')
    return plt.gcf()

def conv_plot(comp, label, grp, n, cycle_end=np.inf):

    # Get data and sort by cycle number
    data = comp.data[n].get_data(label,grp)
    
    data = data[data[:,0].argsort()]
    data = data[data[:,0] < cycle_end,:]

    
    # Get stdev
    std = np.sqrt(comp.data[n].get_var(label,grp))

    plt.figure(figsize=(12,9))
    plt.plot(data[:,0],data[:,1],'k.')

    fom_plot_setup(12,12)
    plt.title(comp.data[n].name)

    plt.axhline(data[-1,1] - std, ls='--', c='k')
    plt.axhline(data[-1,1] + std, ls='--', c='k')

    plt.show()


def get_fom(comparator, label, grp, cycle_caps=[], corr=False):
    x = []
    y = []
    yerr = []

    for data_set in comparator.data:
        # Get FOM data and sort by cycle
        data = data_set.get_data(label,grp,cycle=(not corr))

        #Check if there is a cycle cap
        if cycle_caps:
            for tup in cycle_caps:
                if str(tup[0]) == data_set.name:
                    data = data[data[:,0] < tup[1],:]
        
        data = data[data[:,0].argsort()]

        x.append(float(data_set.name))

        if not corr:
            yerr.append(np.sqrt(data_set.get_var(label,grp)))
        else:
            cycl = data_set.get_data(label, grp, cycle=True)[:,0]
            cycl = cycl[cycl[:].argsort()]
            dcyc = np.average(cyc_cpu_plot(comparator, x[-1], plot=False))
            data[:,1] = np.multiply(data[:,0], data[:,1])
            data[:,1] = np.divide(data[:,1], cycl)
            data[:,1] = data[:,1] * dcyc
            yerr.append(np.sqrt(np.var(data[int(np.ceil(np.shape(data)[0]/2.0)):,1])))
            
        y.append(data[-1,1])
        
    return x, y, yerr

def get_ratios(comparator, label, grp, cycle_caps=[], corr=False):
    x, y, yerr = get_fom(comparator, label, grp, cycle_caps, corr)
    
    # Find base case
    n = x.index(0.1)
    
    r = np.ones_like(x)
    rerr = np.zeros_like(x)
    
    for i in range(0,len(x)):
        if i != n:
            r[i] = y[i]/y[n]
            rerr[i] = r[i]*np.sqrt(np.power(yerr[i]/y[i],2) 
                                   + np.power(yerr[n]/y[n],2))
            
    return x, r, rerr

def plot_title(label, grp, casename):
    if label=='INF_FLX':
        param = ' infinite flux '
    elif label =='INF_TOT':
        param = ' infinite $\Sigma_t$ '

    if grp == 1:
        group = "fast"
    else:
        group = "thermal"
        
    title = casename + param + 'for the ' + group + " group"
    return title

def plot_fom(comparator, casename, label, grp, save=False, fontsize=20, cycle_caps=[], corr=False):
   
    title = plot_title(label, grp, casename)
    
    x, y, yerr = get_fom(comparator, label, grp, cycle_caps = cycle_caps, corr = corr)
    
    plt.figure(figsize=(12,9))
    plt.errorbar(x,y,yerr=yerr, fmt='k.',ms=12)
    plt.title(title)
    fom_plot_setup(fontsize,fontsize)
    plt.xlim([0.05,1.05])
    plt.xticks(np.arange(0.1,1.1, 0.1))
    plt.show()

def plot_ratios(comparator, casename, label, grp, cycle_caps=[], corr=False,
                save=False, fontsize=20, img_dir='~/'):
        
    title = plot_title(label, grp, casename)
    filename = casename.lower() + '_' + label.lower() + '_' + str(grp)

    x, r, rerr = get_ratios(comparator, label, grp, cycle_caps, corr)

    plt.figure(figsize=(12,9))
    plt.errorbar(x,r,yerr=rerr, fmt='k.', ms=12, capsize=0)
    fom_plot_setup(fontsize,fontsize)
    plt.xticks(np.arange(0.1,1.1, 0.1))
    plt.xlim([0.15,1.05])
    plt.title(title)
    plt.ylabel('Normalized FOM')
    plt.xlabel('$t_{\mathrm{wdt}}$', fontsize=fontsize+4)
    plt.axhline(y=1.0, ls='--', c='k')
    if save:
        plt.savefig(img_dir + filename + ".pdf", 
                    format = 'pdf', bbox_inches='tight')
    else:
        plt.show()

def pandas_table(x, y, yerr, r, rerr):
    d = {'twdt' : x, 'fom' : y, 'fom_err': yerr, 'r' : r, 'r_err' : rerr}
    df = pd.DataFrame(d)
    #Move twdt to the front
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    return df

def pandas_format(df, fom_p, rat_p):
    # Formats the data frame
    def fom_form(x):
        return '{:.0f}'.format(np.around(x/np.power(10,fom_p)))
    def ratio(x):
        fs = '{:.' + str(rat_p) + 'f}'
        return fs.format(x)
    df['fom'] = df['fom'].apply(fom_form)
    df['fom_err'] = df['fom_err'].apply(fom_form)
    df['r']  = df['r'].apply(ratio)
    df['r_err']  = df['r_err'].apply(ratio)

def latex(df, fom_p, rat_p):
    pandas_format(df, fom_p, rat_p)
    def f(x): 
        return x
    return df.to_latex(index=False, escape=False, column_format='rrrrr')

def make_table(comp, label, grp, fom_p, rat_p=3, cycle_caps=[]):
    # Get fom and ratios
    x, y, yerr = get_fom(comp, label, grp, cycle_caps)
    x, r, rerr = get_ratios(comp, label, grp, cycle_caps)
    df = pandas_table(x,y,yerr,r,rerr)
    return latex(df, fom_p, rat_p)

def cyc_cpu_plot(comp, twdt, plot=True):
    # Find index
    for i, analyzer in enumerate(comp.data):
        if analyzer.name == str(twdt):
            n = i
    cycles = []
    cpu = []
    for data in comp.data[n].data:
        cycles.append(data.get_data('CYCLE_IDX')[0][0])
        cpu.append(data.get_cpu())
    
    cyc_cpu = np.column_stack((cycles, cpu))
    
    cyc_cpu = cyc_cpu[cyc_cpu[:,0].argsort()]
    dcyc_cpu = []

    for i in range(1,len(cycles)):
        dcyc = cyc_cpu[i,0] - cyc_cpu[i-1,0]
        dcpu = cyc_cpu[i,1] - cyc_cpu[i-1,1]
        dcyc_cpu.append(dcyc/dcpu)

    if plot:
        plt.title('Cycles/CPU time for ' + comp.data[n].name)
        plt.plot(dcyc_cpu, 'k.')
        fom_plot_setup(12,12)
        plt.ylabel('Cycles/CPU time')
        plt.ylim([30,50])
        plt.show()
    else:
        return dcyc_cpu
