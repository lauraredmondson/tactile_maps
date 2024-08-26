'''fmri simulation and analysis
'''
from somatotopic_maps.fmri_map.fmri_simulator import fmri_one_region_rates
from somatotopic_maps.map_plot_analysis.map_and_mesh_plots import sub_plot_panel
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import scipy
import matplotlib as mpl

# In[calculates the map response for 1 stimuli pattern- data = 30x30]

def fmri_map_response_single_trial(weights, input_pattern, **args):
    '''
    fmri map for a single trial.

    Parameters
    ----------
    
    weights: ndarray (size: # variables x # map units)
        map weights between each variable (Eg. afferent and map unit)

    input_pattern: ndarray
        single input pattern
            
    **args : 
        color_map: str
            colormap to use from standard matplotlib library
            (Default: Blues)
            
        region_name: str
              Name of the regions eg.'D1'
            
        plot: bool
              whether to plot (default: False)
            
        ss_a: int
              size of the map along the second axis (default: 30)
            
        ss_b: int
              size of the map along the second axis (default: ss_a)

    Returns
    -------
    activation_map : ndarray (size: ss_a x ss_b)
        Reshaped activation map for stimuli

    '''
    cmap= args.pop('color_map','Blues')
    region_name = args.pop('region_name','') 
    plot = args.get('plot',False)
    ss_a = args.pop('ss_a',30)
    ss_b = args.pop('ss_b',ss_a)
    norm_inputs = args.pop('norm_inputs',True)
    
    # normalise activation of inputs between 0 and 1.
    if norm_inputs:
        input_pattern = (input_pattern - input_pattern.min()) / (input_pattern.max() - input_pattern.min())
    
    # recreate weight matrix
    #w_new = weights*np.matlib.repmat(input_pattern,np.size(weights,1),1).T
    
    # reconstruct map
    #w_new_sum = np.sum(w_new,0)
    
    w_new_sum = np.dot(weights.T, input_pattern)
    
    # reshape and view
    activation_map = np.reshape(w_new_sum,[ss_a,ss_b])

    # plot map
    if plot:
        view_fmri_single_region_map(activation_map, region_name + ' single trial', cmap)
    
    return activation_map

# In[calculates the average map response for 1 region- data = 80x30x30 to 30x30]

def fmri_map_response_all_trials(weights, region_responses, region_name, **args):
    '''
    Mean fmri map over all trials.

    Parameters
    ----------
    weights: ndarray (size: # variables x # map units)
        map weights between each variable (Eg. afferent and map unit)
            
    region_responses: dict (size: no. of trials)
        stimuli and responses for each stimulation.
            
    region_name: str
            Name of the regions eg.'D1'           
            
            
    **args : 
        color_map: str
            colormap to use from standard matplotlib library
            (Default: Blues)
            
        plot: bool
              whether to plot (default: False)
            
        ss_a: int
              size of the map along the second axis (default: 30)
            
        ss_b: 
              size of the map along the second axis (default: 30)

    Returns
    -------
    all_single_maps : ndarray
        array of all the single trial maps for that region
        
    average_map : ndarray
        mean map response for a region

    '''

    # color codes for each grouping variable (eg. fingers on hand)
    cmap = args.pop('color_map','Blues')
    plot = args.get('plot',False)
    ss_a = args.pop('ss_a',30)
    ss_b = args.pop('ss_b',ss_a)
    norm_inputs = args.pop('norm_inputs',True)
    
    # calculate the rates
    input_patterns = fmri_one_region_rates(region_responses)

    # create array to hold all maps (size no.patterns x size of map x size of map)
    all_single_maps = np.zeros((ss_a,ss_b,np.size(input_patterns,1)))
    
    # get map for each trial
    for i in range(np.size(all_single_maps,2)):
        all_single_maps[:,:,i] =fmri_map_response_single_trial(weights, input_patterns[:,i], ss_a=ss_a, ss_b=ss_b, norm_inputs=norm_inputs)
    
    average_map = np.zeros((ss_a,ss_b))
    
    # average over single maps
    average_map = np.mean(all_single_maps,2)

    # plot map
    if plot: 
        view_fmri_single_region_map(average_map,region_name+' all trials', cmap)
        
    return all_single_maps,average_map


# In[calculates the average map response for all regions- data = 20x80x30x30 (example, 30x30 map) to 20x30x30]

def fmri_map_response_all_regions(weights, **args):
    '''
    Calculates responses for each region stimuli. Returns responses to each stimuli
    and average map response for all stimuli placed on that region.

    Parameters
    ----------
    weights: ndarray (size: # variables x # map units)
        map weights between each variable (Eg. afferent and map unit)
            
    **args : 
        stimuli_all: dict (size: no. regions)
            Input patterns for each region and each trial
            
        region_names: list
              list region names eg.'D1'
            
        plot: bool
              whether to plot (default: False)
            
        ss_a: int
              size of the map along the second axis (default: 30)
            
        ss_b: 
              size of the map along the second axis (default: 30)

    Returns
    -------
    all_region_maps : dict (size: no. regions)
        dictionary containing response map for each trial for each region
        
    all_average_maps :  dict (size: no. regions)
        dictionary containing average response map over all trials for each region

    '''
    # pass stimuli_all dictionary
    stimuli_all = args.get('stimuli_all')
    # group names, names for each category eg. D1,D2
    region_names = args.pop('region_names') # should be an array of names
    plot = args.get('plot',False)
    ss_a = args.pop('ss_a',30)
    ss_b = args.pop('ss_b',30)
    norm_inputs = args.pop('norm_inputs',True)
    
    # create array to hold all maps (size no.patterns x size of map x size of map)
    all_region_maps = {}
    all_average_maps = {}
    
    # calculate maps for each trial
    for i in range(len(region_names)):
        # map responses for that region
        [all_region_maps[region_names[i]],all_average_maps[region_names[i]]] =fmri_map_response_all_trials(weights,
            stimuli_all[region_names[i]],region_names[i],ss_a=ss_a,ss_b=ss_b, norm_inputs=norm_inputs)
    
    if plot:
        view_fmri_all_region_maps(all_average_maps,region_names=region_names,cmap='Blues')
        
    return all_region_maps,all_average_maps


# %%[fMRI simulation responses]
 
def fmri_map_responses_zscore(average_maps, **args):
    '''
    Shows the map responses to the stimuli zscored.

    Parameters
    ----------
    average_maps: dict (size: no. regions)
        dictionary containing average response map over all trials for each region
            
    Returns
    -------
    all_zscore_maps : dict (size: no. regions)
        dictionary containing z-scored average response map over all trials for each region
        
    z_maps : ndarray (size: ss_a x ss_b x no.regions)
        z-scored maps as an array

    '''
    ss_a = args.pop('ss_a',30)
    ss_b = args.pop('ss_b',30)
    
    # get region names
    region_names = list(average_maps.keys())
    
    # put all maps into one array
    all_maps = np.zeros((ss_a,ss_b,len(average_maps)))
    
    for i in range(len(average_maps)):
        all_maps[:,:,i] = average_maps[region_names[i]]
        
    # calculate z map of all data
    z_maps = scipy.stats.zscore(all_maps,None)
        
    all_zscore_maps = {}
    
    # add z_score maps back to dictionary
    for i in range(len(average_maps)):
        all_zscore_maps[region_names[i]] = z_maps[:,:,i]

    return all_zscore_maps, z_maps


# In[ 3) Plotting the responses]
    
# In[Plotting code for response map]
def view_fmri_single_region_map(activation_map, region_name, cmap):
    '''
    plots a single map for an fmri stimuli response

    Parameters
    ----------
    activation_map : ndarray (size: ss_a x ss_b)
        Reshaped activation map for stimuli
        
    region_name : str
        region name of the stimuli placement.
        
    cmap: str
        colormap to use from standard matplotlib library
        (Default: Blues)

    Returns
    -------
    None.

    '''
    
    
    fig, ax = plt.subplots()
    
    im = ax.imshow(activation_map,cmap=cmap)
    
    ax.set_title('Map for ' + region_name)
    fig.colorbar(im,ax = ax)
    ax.set_xticks([], [])
    ax.set_yticks([], [])

# In[Plotting code for response map]
# plot all the response maps
def view_fmri_all_region_maps(all_average_maps, region_names, cmap):
    '''
    plots activation map for each region tapping stimuli

    Parameters
    ----------
    all_average_maps : dict (size: no. regions)
        dictionary containing average response map over all trials for each region
        
    region_names: list
        list region names eg.'D1'
        
    cmap: str
        colormap to use from standard matplotlib library
        (Default: Blues)

    Returns
    -------
    None.

    '''
    
    # create colorbar is non-normalised
    vmin = np.zeros(len(region_names))
    vmax = np.zeros(len(region_names))
    
    for i in range(len(region_names)):
        # get map for first region
        vmin[i] = np.min(all_average_maps[region_names[i]])
        vmax[i] = np.max(all_average_maps[region_names[i]])
    
    vmin = np.min(vmin)
    vmax = np.max(vmax)
    
    # run subplot panelling- get back the two numbers for x and y panel size
    panel = sub_plot_panel(len(region_names),space_h = 0.7, space_w = 0.7)
    
    # create figure
    fig = panel[0]
    axes = panel[1]
    
    fig.suptitle('Region maps')
    
    # create map for each region and add
    for i in range(len(region_names)):
        # get map for first region
        sheet_final = all_average_maps[region_names[i]]
        
        # show map
        c = axes[i].imshow(sheet_final,cmap=plt.get_cmap(cmap), vmin=vmin, vmax=vmax)
        
        # set title for figure
        axes[i].set_title('Cortical map for ' + region_names[i])
        
        # turn off the axis box around the map
        axes[i].axis('off')
    
    # add colorbar for all
    cbar = fig.colorbar(c, ax=axes)
    cbar.set_label('Cortical map activation', rotation=90)
    

# %%[]
def fmri_WTA_map(all_average_maps, cmap, threshold = 0):
    '''
    Plots the winner takes all map for the digit tapping

    Parameters
    ----------
    all_average_maps : dict with 5 keys
        Average map responses for each digit tapping
        
    cmap : Numpy array of size- number of digits X 3 (RGB values)
        Colormap RGB values for each digit

    Returns
    -------
    None.

    '''
    
    # get dictionary keys
    dict_keys = list(all_average_maps.keys())
    
    map_size_x = np.size(all_average_maps[dict_keys[0]],0)
    map_size_y = np.size(all_average_maps[dict_keys[0]],1)
    
    # transfer dict values to array:
    activation_array = np.zeros((map_size_x ,map_size_y, len(dict_keys)))
    
    for digit in range(len(dict_keys)):
        activation_array[:,:,digit] = all_average_maps[dict_keys[digit]]
        

    # find index of max unit
    activation_array[activation_array < threshold] = 0
    
    # find all where equal to zero
    zero_idx = np.where(np.sum(activation_array,2) == 0)
    print(len(zero_idx[0]))
    
    wta_index = np.argmax(activation_array,axis=2)
    
    cmap_section = cmap[:5,:]
    
    if len(zero_idx[0]) > 0:
        wta_index[zero_idx[0],zero_idx[1]] = np.size(activation_array,2)
        
        region_list = dict_keys.copy()
        region_list.append('Null')

        cmap_section = np.vstack([cmap_section, [.5,.5,.5]]) 

    else:
        region_list = dict_keys
    
    # create colorbar
    new_cmap = mpl.colors.ListedColormap(cmap_section)
    
    # create figure
    fig, ax = plt.subplots()

    # add map
    ax.imshow(wta_index,cmap=new_cmap)
    
    ax.set_title('WTA map for digit tapping')
    
    # add location of colorbar
    cbaxes = fig.add_axes([0.85, 0.1, 0.03, 0.8]) 
    
    # add bounds for the colors
    bounds = np.linspace(0,1,len(region_list)+1).tolist()
    
    dist = (1/len(region_list))/2
    # create custom colorbar
    cb = mpl.colorbar.ColorbarBase(ax = cbaxes, cmap=new_cmap,
                                    boundaries=bounds,
                                    ticks=np.array(bounds)[1:]-dist,
                                    orientation='vertical')

    # add the tick labels
    cb.ax.set_yticklabels(region_list) 
    
    # add the colorbar label
    cb.set_label('Digit tapped')
    
    # turn off the axis box around the map
    ax.axis('off')

    

