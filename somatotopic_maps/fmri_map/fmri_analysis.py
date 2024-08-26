# fMRI style of analysis- analysis on activated weight patterns, rather than on all weights.
import numpy as np
import copy
from somatotopic_maps.map_plot_analysis.map_analysis import find_peak

# %%[fMRI region sizes]

def fmri_region_sizes(map_data, **args):
    '''
    region sizes for the map from the fmri simulation.
    load average map data for each map/ 'participant'
    
    Parameters
    ----------
    map_data : ndarray (size: weights x no afferents x number_of_stimuli)
        number of stimuli is the digit activations.
        array with all the map weights in for each digit.
        
        
    **args :
        z_score: float
            cut off for map activations

    Returns
    -------
    area_sizes : ndarray
        area sizes for each digit/ region
        
    area_sizes_per : ndarray
        percentage area sizes for each digit/ region

    '''
    z_score = args.get('z_score_thresh',3)
    
    # get map keys from dict
    key_list = list(map_data.keys())
    
    # make array to add size data to
    area_sizes = np.zeros((len(key_list)))
    
    # make array to add size data to
    area_sizes_per = np.zeros((len(key_list)))
    
    # count the number of voxels for each representation
    for i in range(len(key_list)):
        
        # get the first map
        map_c = map_data[key_list[i]]
        
        # count number of units above z-thresh
        area_sizes[i] = np.sum(map_c > z_score)
        
        area_sizes_per[i] = area_sizes[i]/(np.size(map_c,0)*np.size(map_c,1))*100
        
    # return the size data
    return area_sizes, area_sizes_per


# %%[calculate overlaps]

def fmri_cortical_overlaps(map_data, **args):
    '''
    Calculates the overlaps
    returns: thresholded map for each group/ category/ region
    area counts- number of units in each group
    indexed locations of each unit (both in 2d and 1d locations)
    threshold is between 0 and 1

    Parameters
    ----------
    map_data : ndarray (size: weights x no afferents x number_of_stimuli)
        number of stimuli is the digit activations.
        array with all the map weights in for each digit.
        
    **args : 
        threshold: float (default: 3)
            cut off for map activations

    Returns
    -------
    index_list : ndarray (size: no.map units x no.regions)
        each column is list of map unit positions where that map activation is over
        a threshold.
        
    counts : ndarray (size: no.regions)
        number of units over threshold for each region.
        
    w_all : ndarray (size: ss_a x ss_b x no.regions)
        weights over the threshold as maps for each region stimuli. Units under threshold
        set to zero.

    '''
    threshold = args.pop('threshold',3)    


    key_list = list(map_data.keys())
    example_map = map_data[key_list[0]]
    index_list = np.empty((np.size(example_map,0)*np.size(example_map,1),len(key_list)))
    index_list[:] = np.nan
    
    counts = np.zeros((len(key_list)))
    
    w_all = np.zeros((np.size(example_map,0),np.size(example_map,1),len(key_list)))
    
    map_data_overlap = copy.deepcopy(map_data)
    
    # for each area keep values above a certain weight theshold. 
    
    # get only the weight matrix for that region
    for i in range(len(key_list)):
        
        weights_thresh = map_data_overlap[key_list[i]]
    
        weights_thresh[(weights_thresh < threshold)] = 0
        
        # add to 3d array
        w_all[:,:,i] = weights_thresh
        
        weights_thresh = np.reshape(weights_thresh,[1,np.size(example_map,0)*np.size(example_map,1)])
        
        counts[i] = np.count_nonzero(weights_thresh)
        
        indexes = np.nonzero(weights_thresh)[1]
        
        index_list[0:len(indexes),i] = indexes
    
    
    return index_list, counts, w_all

# %%[fMRI peak distances]

def fmri_peak_map_coords(map_data, group_name):
    '''
    Find the peak to peak distances in the fMRI patterns
    Calculates peak for a map matrix for each region

    Parameters
    ----------
    map_data : ndarray (size: weights x no afferents x number_of_stimuli)
        number of stimuli is the digit activations.
        array with all the map weights in for each digit.
        
    group_name : list (size: no. regions)
        names of each region

    Returns
    -------
    peak_coords : ndarray (size: no. regions x 2)
        peak map activation coordinate on the map for each region/ stimuli
        
    max_weights : ndarray (size: no.regions)
        Corresponding weight activations for max unit

    '''
    
    peak_coords = np.zeros((len(group_name),2))
    
    max_weights = np.zeros((len(group_name)))
        
    # loop through key_list
    for i in range(len(group_name)):
        
        # find the coordinates unit with the strongest weight
        peak_coords[i,:],max_weights[i] = find_peak(map_data[group_name[i]])

    # return coordinates and weight
    return peak_coords, max_weights


