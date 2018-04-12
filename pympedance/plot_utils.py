import numpy as np
import matplotlib.pyplot as pl


def bodeplot(f, h, xscale='lin', yscale='db', ax=None, **kwargs):
    """
    Plot a complex-valued function h(f) in module/ phase
    pair of axes
    
    Arguments:
    * f: frequency values for h
    * h: complex-valued function of f
    * xscale ('lin', 'log' or 'oct')
    * yscale ('lin', 'log', 'db' or  'dbpow')
    """
    if ax is None:
        fig, ax = pl.subplots(2,sharex=True)
        
    else:
        fig = ax[0].figure
    

    if xscale == 'lin' or xscale == 'log' :
        x = f
        ax[1].set_xlabel('Frequency (Hz)')
    elif xscale == 'oct':
        x = np.log2(f)
        ax[1].set_xlabel('Octaves')
    if yscale == 'lin' or yscale == 'log':
        yabs = np.abs(h) 
        yarg = np.angle(h)
        ax[0].set_ylabel('Magnitude (lin)')
    elif yscale == 'db':
        yabs = 20*np.log10(np.abs(h))
        yarg = np.angle(h)
        ax[0].set_ylabel('Magnitude (dB)')
    elif yscale == 'dbpow':
        yabs = 10*np.log10(np.abs(h))
        yarg = np.angle(h)
        ax[0].set_ylabel('Magnitude (dB)')
    ax[1].set_ylabel('Phase (rad)')


    ax[0].plot(x,yabs,**kwargs)
    ax[1].plot(f,yarg,**kwargs)
    
    if xscale == 'log':
        ax[0].set_xscale('log')
        ax[1].set_xscale('log')
    if yscale == 'log':
        ax[0].set_yscale('log')
    
    yrange = np.diff(ax[1].get_ylim())
    npi = yrange/np.pi
    pifrac = 2*np.pi/yrange
    
    mult = 1
    div = 2
    #print(mult,div)
    
    def format_func(value, tick_number):
        # find number of multiples of pi/2
        N = int(np.round(div * value / np.pi))
        if N == 0:
            return "0"
        elif N == div:
            return r"$\pi$"
        elif N == -div:
            return r"$-\pi$"
        elif N == 1:
            return r"$\pi/{}$".format(div)
        elif N == -1:
            return r"$-\pi/{}$".format(div)

        elif N % div > 0:
            return r"${0}\pi/{1}$".format(N,div)
        else:
            return r"${0}\pi$".format(N // div)
        
    ax[1].yaxis.set_minor_locator(pl.MultipleLocator(mult*np.pi/div/8))
    ax[1].yaxis.set_major_locator(pl.MultipleLocator(mult*np.pi/div))
    ax[1].yaxis.set_major_formatter(pl.FuncFormatter(format_func))

    return fig, ax
