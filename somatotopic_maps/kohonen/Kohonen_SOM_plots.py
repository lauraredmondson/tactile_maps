import numpy as np
import matplotlib.pyplot as plt
import scipy

# %%
def plot_kohonen_parameters(learning_rate,neighbourhood_size):
    '''
    plots the learning rate and neighbourhood parameters

    Parameters
    ----------
    learning_rate : ndarray
        learning rate over time/ input patterns
        
    neighbourhood_size : ndarray
        neighbourhood sizes over time/ input patterns

    Returns
    -------
    None.

    '''
    
    # number of timesteps
    t = np.linspace(0,len(learning_rate),len(learning_rate))
    
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('SOM inputs')
    ax1.set_ylabel('Learning rate', color=color)
    ax1.plot(t, learning_rate, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    color = 'tab:blue'
    ax2.set_ylabel('Neighbourhood size', color=color)  # we already handled the x-label with ax1
    ax2.plot(t, neighbourhood_size, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
 
 
# %%   
def make_patch_spines_invisible(ax):
    '''
    makes the spines invisible for an axis

    Parameters
    ----------
    ax : matplotlib axis
        axis for the plot

    Returns
    -------
    None.

    '''
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
 
# %%
def plot_all_kohonen_parameters(learning_rate,neighbourhood_size,weights_decay,**args):
    '''
    plots all the learning parameters on one graph 

    Parameters
    ----------
    learning_rate : ndarray
        learning rate over time/ input patterns
        
    neighbourhood_size : ndarray
        neighbourhood sizes over time/ input patterns

    weights_decay : ndarray
        weight changes in the model, average weight changes per unit
        
    **args : 
        curve_fit: bool
            Fits a curve through the weight changes (default: False)


    Returns
    -------
    None.

    '''
    curve_fit = args.pop('curve_fit',False)
    logscale = args.pop('logscale',False)
    
    fig, ax = plt.subplots(figsize=(10,5))
    fig.subplots_adjust(right=0.75)
    
    par1 = ax.twinx()
    par2 = ax.twinx()
    
    # Offset the right spine of par2.  The ticks and label have already been
    # placed on the right by twinx above.
    par2.spines['right'].set_position(("axes", 1.2))
    
    # Having been created by twinx, par2 has its frame off, so the line of its
    # detached spine is invisible.  First, activate the frame but make the patch
    # and spines invisible.
    make_patch_spines_invisible(par2)
    
    # Second, show the right spine.
    par2.spines['right'].set_visible(True)
        
    # number of timesteps
    t = np.linspace(0,len(learning_rate),len(learning_rate))
    
    # best fit line for weights decay
    if curve_fit:
        w_param = scipy.optimize.curve_fit(lambda p,a,b: a*np.exp(-b*p),  t,   weights_decay)
        weights_decay = w_param[0][0]*np.exp(-(w_param[0][1])*t)
        
    p1, = ax.plot(t,  weights_decay, "g-", label='Weight change')
    p2, = par1.plot(t, neighbourhood_size, "r-", label='Neighbourhood size')
    p3, = par2.plot(t, learning_rate, "b-", label='Learning rate')
    
    ax.set_xlabel('Input patterns')
    ax.set_ylabel('Average weight change per map unit')
    par1.set_ylabel('Neighbourhood size')
    par2.set_ylabel('Learning rate')
    
    ax.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())
    par2.yaxis.label.set_color(p3.get_color())
    
    tkw = dict(size=4, width=1.5)
    ax.tick_params(axis='y', colors=p1.get_color(), **tkw)
    par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
    ax.tick_params(axis='x', **tkw)
    
    lines = [p1, p2, p3]
    
    ax.legend(lines, [l.get_label() for l in lines])
    
    if logscale:
        ax.set_xscale('log')
    plt.show()

    