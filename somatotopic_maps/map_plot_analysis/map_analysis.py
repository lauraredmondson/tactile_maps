'''Hand plot analysis
    Includes code to:
      WTA calculator:
          -recalculates weight matrix to be WTA
      Area sizes
        - Calculate area sizes as WTA (2 different ways)
        - Calculate area sizes when a threshold is included (overlaps allowed)
        - Plot area sizes by main or sub hand regions
      Overlaps
        - Calculate overlap of any two regions given, returns DICE and 
        - Calculate overlap for all hand regions (main) returns as array of DICE     
            and array of overlap indexes in the map
      Hand mesh
        - Calculates the distance along the hand mesh of any two given points
      Distances
        - Calculates the distance between fingers at each of the phalanges. 
              Returns array of distances for each phalange eg. tip, medial, proximal
'''

from somatotopic_maps.hand_model.hand_mesh import get_connections_list
from .map_and_mesh_plots import plot_map, view_map, view_map_all_regions
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import spatial, ndimage
from scipy.spatial.distance import pdist, squareform
from matplotlib.collections import LineCollection
from itertools import combinations


# %%[Region weight matrices]

def region_weight_matrices(weights, hand_data, **args):
    """
    returns weight matrices for each part set in th grouping variable, 
    as dictionary, returns reshaped maps for each region

    Parameters
    ----------
    weights: ndarray (size: # variables x # map units)
        map weights between each variable (Eg. afferent and map unit)
    
    hand_data : Dict
        Hand region information
        
    **args :
        ss_a : int
            sheet size in a dimension (default: sqrt map size)
        
        ss_b : int
            sheet size in b dimension (default: ss_a)

    Returns
    -------
    separate_weights : dict (size: no of groups)
        separate weight matrices for afferents from each region/ group
        
    reshaped_maps_dict : dict (size: no of groups)
        separate reshaped weight matrices (maps) for afferents from each region/ group
        each map is size ss_a x ss_n
        
    separate_maps : ndarray (size: (ss_a x ss_b) x no. of groups)
        Maps from each group/ region squashed into one array. Each row is a different map
        
    reshaped_maps : ndarray (ss_a x ss_b x no. of groups)
        maps from each region, reshaped and stacked on top of each other (3D array)

    """
    ss_a = args.get('ss_a', int(np.sqrt(np.size(weights,0))))
    ss_b = args.get('ss_b',ss_a)
    
    group_name = hand_data['region_list']
    grouping_variable = hand_data['region_index']
    
    # set up dictionary
    separate_weights = {}
    
    reshaped_maps_dict = {}
    
    reshaped_maps = np.zeros((ss_a,ss_b,len(group_name)))
    
    separate_maps = np.zeros((ss_a*ss_b,len(group_name)))
    
    for i in range(len(group_name)):
        
        idx = np.where(grouping_variable == i)[0]
        
        separate_weights[group_name[i]] = weights[idx,:]
        
        separate_maps[:,i] = np.sum(weights[idx,:],0)
        
        reshaped_maps[:,:,i] = np.reshape(separate_maps[:,i],[ss_a,ss_b])
        
        reshaped_maps_dict[group_name[i]] = np.reshape(separate_maps[:,i],[ss_a,ss_b])
        
    return separate_weights, reshaped_maps_dict, separate_maps, reshaped_maps


# %%[Peak to peak distances]

def find_peak(active_data):
    """
    Calculates peak for a map matrix for each region

    Parameters
    ----------
    active_data : ndarray (size: ss_a x ss_b)
        active map

    Returns
    -------
    peak_coord : ndarray (size: 2)
        maximally active unit coordinate
        
    weight : float
        corresponding weight from the maximally active unit

    """
    
    max_unit = np.unravel_index(active_data.argmax(), active_data.shape)
    
    peak_coord = np.array((max_unit[0],max_unit[1]))
        
    weight = np.max(active_data)
    
    # return coordinates and weight
    return peak_coord, weight


  # %%[Peak to peak distances]
 
def peak_map_coords(weights, hand_data, **args):
    """
    calculates peak for a map matrix for each region

    Parameters
    ----------
    weights: ndarray (size: # variables x # map units)
        map weights between each variable (Eg. afferent and map unit)
        
    hand_data : Dict
        Hand region information
        
    **args :
        ss_a : int
            sheet size in a dimension (default: sqrt map size)
        
        ss_b : int
            sheet size in b dimension (default: ss_a)

    Returns
    -------
    peak_coords : ndarray (size: no. of groups x 2)
        maximally active unit coordinate for each group
        
    max_weights : ndarray (size: no. of groups)
        corresponding weight from the maximally active unit for each group

    """
    ss_a = args.pop('ss_a', int(np.sqrt(np.size(weights,0))))
    ss_b = args.pop('ss_b', ss_a)

    group_name = hand_data['region_list']
    
    peak_coords = np.zeros((len(group_name),2))
    
    max_weights = np.zeros((len(group_name)))
        
    region_maps = region_weight_matrices(weights, hand_data, ss_a=ss_a, ss_b=ss_b)[1]
            
    # loop through group_name list
    for i in range(len(group_name)):
    
        # find the coordinates unit with the strongest weight
        peak_coords[i,:],max_weights[i] = find_peak(region_maps[group_name[i]])

    # return coordinates and weight
    return peak_coords, max_weights
    
# %%[Peak to peak distances]

def peak_map_distance_pair(coords):
    """
    calculates the distances of the peak coordinates for two regions

    Parameters
    ----------
    coords : ndarray
        peak coordinates for two region    

    Returns
    -------
    ndarray of distance between the two coordinates

    """
    
    return pdist(coords)

# %%[Peak to peak distances]

def peak_map_distances(coords, group_name, **args):
    """
    Calculates the distances of the peak coordinates

    Parameters
    ----------
    coords : ndarray (size: no. of groups x 2)
        coords of the activation peak in the map
        
    group_name : list (size: no. of groups)
        groups of variables/ afferents in the map eg. regions ['D1', 'D2']
        Must match order in weights_regions
        
    **args :
        plot: bool
            whether to plot the map (default: False)

    Returns
    -------
    distances_full : ndarray (size: no. of groups x no. of groups)
        distance matrix between each group and all others

    """    
    plot = args.get('plot',False)
    
    r_key = {}
    
    for i in range(len(group_name)):
        r_key[group_name[i]] = i

    # use itertools combinations to obtain all pairs
    all_pairs = list(combinations(group_name,2))

    # add self value
    for i in range(len(group_name)):
        all_pairs.append((group_name[i],group_name[i]))
        
    # store dissimilarities
    distances_full = np.zeros((len(group_name),len(group_name)))
    
    # loop through pair list and find distance
    for i in range(len(all_pairs)):
        # find location of first in pair
        pair1 = group_name.index(all_pairs[i][0])
        
        # find loc of second in pair
        pair2 = group_name.index(all_pairs[i][1])
        
        diss_value = peak_map_distance_pair([coords[pair1],coords[pair2]])
        distances_full[r_key[all_pairs[i][0]],r_key[all_pairs[i][1]]] = diss_value
        distances_full[r_key[all_pairs[i][1]],r_key[all_pairs[i][0]]] = diss_value
    
    if plot:
        fig, ax = plt.subplots()
        im = ax.imshow(distances_full,cmap='Blues')
        ax.set_title('Distances_full')
        ax.set_xticks(range(len(group_name)))
        ax.set_yticks(range(len(group_name)))
        ax.set_xticklabels(group_name)
        ax.set_yticklabels(group_name)
        fig.colorbar(im,ax = ax)
    
    
    # return matrix
    return distances_full

# %%[WTA weight matrix]
   
def wta_weights(weights, hand_data, **args):
    """
    recalculates the weight matrix to be WTA
    gv_index should be 'main' for the hand if calculating WTA matrix for
    map.
    Returns w_wta, which is the updated w array
    
    
    WTA method 1 is just the biggest weight
    WTA method 2 is the largest mean weight from each region
    WTA method 3 is the sum of the weights from each region
    returns the group index of the max value

    Parameters
    ----------
    weights: ndarray (size: # variables x # map units)
        map weights between each variable (Eg. afferent and map unit)

    hand_data : Dict
        Hand region information
        
    **args :
        method: str
            method to calculate wta map from 'wta_1', 'wta_2' and 'wta_3'
            (default: 'wta_1')

    Returns
    -------
    weights_wta : ndarray (size: # variables x # map units)
        new weight matrix based on summary method
        
    max_group : ndarray (size: # variables x 1)
        index of group name assigned as wta to each map unit

    """

    method = args.get('method','wta_1')
    
    group_index = hand_data['region_index']
    group_name = hand_data['region_list']
    
    weights_wta = np.zeros((np.size(weights,0),np.size(weights,1)))
    
    # make w local
    weights_old = weights.copy()
    
    max_group = np.zeros((np.size(weights,1)+1,1))
    
    # find mean weights for method 2
    if method in ['wta_2','wta_3']:
        
        for i in range(len(group_name)):
            idx = np.where(group_index == i)[0]
            
            for j in range(np.size(weights_wta,1)):
            # add this mean to all the indexes in group
                if method == 'wta_2':
                    # method 2- calculate mean of each group
                    weights_old[idx,j] =  np.mean(weights_old[idx,j])
                else: 
                    # method 3- calculate summed weights of each group
                    weights_old[idx,j] =  np.sum(weights_old[idx,j])
               
    # method 1- find the largest unit, set everything else to zero
    for i in range(np.size(weights_wta,1)): 
        
        max_unit = np.argmax(weights_old[:,i])
        
        # put value of max_unit in array
        weights_wta[max_unit,i] = weights_old[max_unit,i]
        
        # get index of this max unit       
        max_group[i] = group_index[max_unit]

    # return w_new
    return weights_wta, max_group

# # %% Threshold wta

# def wta_weights_threshold(weights, group_index, group_name, threshold, **args):
#     """
#     recalculates the weight matrix to be WTA
#     gv_index should be 'main' for the hand if calculating WTA matrix for
#     map.
#     Returns w_wta, which is the updated w array
    
    
#     WTA method 1 is just the biggest weight
#     WTA method 2 is the largest mean weight from each region
#     WTA method 3 is the sum of the weights from each region
#     returns the group index of the max value

#     Parameters
#     ----------
#     weights: ndarray (size: # variables x # map units)
#         map weights between each variable (Eg. afferent and map unit)

#     group_index: ndarray (size: # variables x 1)
#         corresponding index of the group each variable/ afferent is in. 
#         eg. [0] in group ['D1']
        
#     group_name: list (size: no. of groups)
#         groups of variables/ afferents in the map eg. regions ['D1', 'D2']

#     **args :
#         method: str
#             method to calculate wta map from 'wta_1', 'wta_2' and 'wta_3'
#             (default: 'wta_1')

#     Returns
#     -------
#     weights_wta : ndarray (size: # variables x # map units)
#         new weight matrix based on summary method
        
#     max_group : ndarray (size: # variables x 1)
#         index of group name assigned as wta to each map unit

#     """

#     method = args.get('method','wta_1')
    
#     weights_wta = np.zeros((np.size(weights,0),np.size(weights,1)))
    
#     # make w local
#     weights_old = weights.copy()
    
#     max_group = np.zeros((np.size(weights,1),1))
    
#     # find mean weights for method 2
#     if method in ['wta_2','wta_3']:
        
#         for i in range(len(group_name)):
#             idx = np.where(group_index == i)[0]
            
#             for j in range(np.size(weights_wta,1)):
#             # add this mean to all the indexes in group
#                 if method == 'wta_2':
#                     # method 2- calculate mean of each group
#                     weights_old[idx,j] =  np.mean(weights_old[idx,j])
#                 else: 
#                     # method 3- calculate summed weights of each group
#                     weights_old[idx,j] =  np.sum(weights_old[idx,j])
               
                
#     # find index of max unit
#     weights_old[weights_old < threshold] = 0
    
#     # find all where equal to zero
#     zero_idx = np.where(np.sum(weights_old,0) == 0)[0]
#     print(len(zero_idx))
            
#     # method 1- find the largest unit, set everything else to zero
#     for i in range(np.size(weights_wta,1)): 
        
#         # find index of max unit
#         max_unit = np.argmax(weights_old[:,i])
        
#         # put value of max_unit in array
#         weights_wta[max_unit,i] = weights_old[max_unit,i]
        
#         # get index of this max unit       
#         max_group[i] = group_index[max_unit]

#     weights_wta[:,zero_idx] = len(group_name)
#     max_group[zero_idx] = len(group_name)

#     # return w_new
#     return weights_wta, max_group, zero_idx
    

 # %%[calculate area sizes]

def area_size(weights, hand_data, **args):
    """
    Calculates area sizes based on WTA method, or a threshold
    if threshold is not given, calculates based on a WTA method.
    If other WTA method given uses alternative.
    If calculating area sizes for hand sub region, ensure the grouping variable 
    is for sub regions

    Parameters
    ----------
    weights: ndarray (size: # variables x # map units)
        map weights between each variable (Eg. afferent and map unit)

    hand_data : Dict
        Hand region information
        
    **args :
        method: str
            method to calculate wta map from 'wta_1', 'wta_2' and 'wta_3'
            (default: 'wta_1')

    Returns
    -------
    unique_counts : ndarray (size: length group_name)
        Array with number of wta map units assigned to each group.
        Ordering corresponds to group_name

    """
    method = args.pop('method','wta_1')

    _ , weights_idx = wta_weights(weights, hand_data, method=method)
    unique_weights_idx = np.unique(weights_idx,return_counts=True)
    
    unique_counts = np.zeros((len(hand_data['region_list']))) 
    
    # put unique counts in an array
    for i in range(len(unique_weights_idx[0])):
        unique_counts[int(unique_weights_idx[0][i])] = unique_weights_idx[1][i]

    return unique_counts


# %%[calculate area sizes]
 
def plot_area_size(counts, hand_data, **args):
    """
    plots the area sizes for each of the hand groups
    counts is number of cortical units in each group
    cmap is list of color codes for each grouping variable
    if thresholded area size is wanted, run cortical_overlaps to get counts

    Parameters
    ----------
    counts : ndarray (size: length group_name)
        Array with number of wta map units assigned to each group.
        Ordering corresponds to group_name
        
    hand_data : Dict
        Hand region information
        
    **args :
        plot_type: str
            plotting type required, percentages or raw data (default: 'per')
            change to '' for raw data
            
    Returns
    -------
    None.

    """
    plot_type = args.get('plot_type','per')
    group_name = args.get('group_name', hand_data['region_list'])
    
    #group_name = hand_data['region_list']
    cmap = hand_data['rgb_all']
    
    # set labels
    y_axis_label = 'Number of units'
    
    # calculate percentage of areas- plot as percentage, update axis labels
    if plot_type == 'per':
        counts = counts/np.sum(counts)*100
        # update axis label
        y_axis_label = 'Percentage of units'

    y_pos = np.arange(len(group_name))

    fig, ax = plt.subplots()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.bar(y_pos, counts, align='center', color=cmap, alpha=0.5)
    if len(group_name) > 10:
        plt.xticks(y_pos, group_name, rotation=45)
    else:
        plt.xticks(y_pos, group_name)
    plt.ylabel(y_axis_label)
    plt.title('Cortical area sizes')
    plt.tight_layout()

# %%[plots area_sizes for all three WTA methods]    

def WTA_method_compare(counts_all, hand_data, **args):
    """
    counts should be an array

    Parameters
    ----------
    counts_all : ndarray (size: no. groups x 4)
        contains the three WTA methods to compare, and percentage sizes of the hand regions
        ordering: [wta1, wta2, wta3, hand_sizes]
        
    hand_data : Dict
        Hand region information
        
    **args :
        plot_type: str
            plotting type required, percentages or raw data (default: 'per')
            change to '' for raw data

    Returns
    -------
    None.

    """
    plot_type = args.get('plot_type','per')

    group_name = hand_data['region_list']
    
    # set labels
    y_axis_label = 'Number of units'
    
    # calculate percentage of areas- plot as percentage, update axis labels
    if plot_type == 'per':
        counts_all /= counts_all.sum(axis=0, keepdims=True)
        counts_all *= 100
        # update axis label
        y_axis_label = 'Percentage of units'
    
    width = 0.35
    
    y_pos = np.linspace(0,len(group_name)+len(group_name)*width,len(group_name))
    
    fig, ax = plt.subplots()
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    p1 = ax.bar(y_pos, counts_all[:,0], width, color='tab:blue',bottom=0)
    p2 = ax.bar(y_pos+width,  counts_all[:,1], width, color='tab:orange', bottom=0)
    p3 = ax.bar(y_pos+width*2 ,  counts_all[:,2], width, color='tab:green', bottom=0)
    p4 = ax.bar(y_pos+width*3 ,  counts_all[:,3], width, color='tab:red', bottom=0)
    
    ax.legend((p1[0], p2[0], p3[0], p4[0]), ('WTA 1', 'WTA 2', 'WTA 3', 'Hand area sizes'))
    plt.xticks(y_pos, group_name)
    plt.ylabel(y_axis_label)
    ax.set_xticks(y_pos + width, group_name)
    plt.title('Cortical area sizes')
    ax.autoscale_view()


    
# %%[calculate overlaps]

def cortical_overlaps(weights, hand_data, **args):
    """
    calculates the overlaps
    returns: thresholded map for each group/ category/ region
    area counts- number of units in each group
    indexed locations of each unit (both in 2d and 1d locations)
    threshold is between 0 and 1

    Parameters
    ----------
    weights: ndarray (size: # variables x # map units)
        map weights between each variable (Eg. afferent and map unit)

    hand_data : Dict
        Hand region information

    **args :
        ss_a : int
            sheet size in a dimension (default: sqrt map size)
        
        ss_b : int
            sheet size in b dimension (default: ss_a)
            
        threshold: float
            sets threshold on map activation to be included in overlaps
            (default: 1)

    Returns
    -------
    index_list : ndarray (size: ss_a*ss_b (# map units) x # groups)
        All map unit indexes where weights from that region are greater than the
        threshold given
        
    counts : ndarray (size: length group_name)
        Array with number of wta map units assigned to each group.
        Ordering corresponds to group_name
        
    weights_all : ndarray (size: ss_a x ss_b x len(group_name))
        map weights for each region, reshaped for map ploting

    """
    ss_a = args.pop('ss_a', int(np.sqrt(np.size(weights,0))))
    ss_b = args.pop('ss_b', ss_a)
    threshold = args.pop('threshold',1)    
    
    group_index = hand_data['region_index']
    group_name = hand_data['region_list']
    
    # check inputs
    assert np.size(weights,1) == ss_a*ss_b, 'Number of map units and map sizes do not match, check ss_a and ss_b'
    assert len(group_index) == np.size(weights,0), 'Number of groups_indexes and variables in weights do not match'

    # create 3D numpy array for weight matrices of each group
    weights_all = np.zeros((ss_a,ss_b,len(group_name)))
    
    index_list = np.empty((np.size(weights,1),len(group_name)))
    index_list[:] = np.nan
    
    counts = np.zeros((len(group_name),1))
    
    # normalise w
    # rescale W to 0 to 1 values
    norm_w = weights/np.sum(weights,axis=0)
    
    # for each area keep values above a certain weight theshold. 
    
    # get only the weight matrix for that region
    for i in range(len(group_name)):
    
        idx = np.where(group_index == i)
        
        #w_new =  np.mean(norm_w[idx[0],:],0)
        weights_new = np.sum(norm_w[idx[0],:],0)
        
        if threshold == 1:
            weights_new[(weights_new < np.max(weights_new))] = 0 # shows most active unit
        else:
            weights_new[(weights_new < threshold)] = 0
        
        # add to 3d array
        weights_all[:,:,i] = np.reshape(weights_new,[ss_a,ss_b])
        
        counts[i] = np.count_nonzero(weights_all[:,:,i])
        
        indexes = np.nonzero(weights_new)[0]
        
        index_list[0:len(indexes),i] = indexes
    
    counts = np.squeeze(counts)
    
    return index_list, counts, weights_all

    
# %%[Dice coefficient]
  
def dice_coeff(index_list_a, index_list_b):
    """
    calculates the dice coefficient for two regions
    
    Parameters
    ----------
    index_list_a : ndarray (size: # map units)
        mpa unit indexes over threshold for a single group/region
        
    index_list_b : ndarray (size: # map units)
        mpa unit indexes over threshold for a second single group/region

    Returns
    -------
    dice : float
        Dice coeffcient for two regions

    """
    
    # calculate dice coefficient
    intersection = len(np.intersect1d(index_list_a,index_list_b))
    dice = (2*intersection)/(sum(~np.isnan(index_list_a))+sum(~np.isnan(index_list_b)))
    
    # return dice value
    return dice

    
# %%[all Dice]

def dice_all(index_list_all, group_name, **args):
    """
    calculates the dice coefficient for all possible pairs of the grouping
    variable. Returns as numpy array of N*N ( where N is number of items in grouping 
    variable)

    Parameters
    ----------
    index_list_all : ndarray (size: # map units x # groups)
        Indexes of the map units where a group/ regions weights are greater than a
        threshold (calulated with cortical_overlaps() )
        
    group_name: list (size: no. of groups)
        groups of variables/ afferents in the map eg. regions ['D1', 'D2']

    **args :
        plot: bool
            whether to plot the dice maps (default: False)

    Returns
    -------
    dice_df : pandas dataframe
        dataframe with dice coefficients

    """
    plot = args.pop('plot',False)
    
    # create dataframe- for seaborn plotting heatmap
    dice_coeff_all = np.zeros((np.size(index_list_all,1),np.size(index_list_all,1)))
    
    # make column names
    # rempmat group names and reshape
    col_1 = np.tile(group_name,(len(group_name),1))
    col_1 = np.reshape(np.transpose(col_1),[len(group_name)**2])
    # repmat group names
    col_2 = np.tile(group_name,len(group_name))

    dice_df = pd.DataFrame(index=np.arange(0,len(col_1)),columns=['group_1','group_2','dice'])
    
    dice_df['group_1'] = col_1
    dice_df['group_2'] = col_2
    
    # calculate all possible dice combinations
    for i in range(np.size(index_list_all,1)):
                    
        for j in range(np.size(index_list_all,1)):
            
            if i == j:
                dice_coeff_all[i,j] = 1
            
            else:
                dice_coeff_all[i,j] = dice_coeff(index_list_all[:,i], index_list_all[:,j])
            
    dice_df['dice'] = np.reshape(dice_coeff_all,[len(group_name)**2]) 
            
    dice_df = dice_df.pivot(index="group_1", columns="group_2", values="dice")
    
    if plot == True:
        plot_dice(dice_df)
 

    return dice_df

# %%[plots the Dice coefficient]

def plot_dice(dice_df):
    """
    plots the Dice coefficient values as a heatmap

    Parameters
    ----------
    dice_df : pandas dataframe
        dataframe with dice coefficients

    Returns
    -------
    None.

    """

    fig, ax = plt.subplots(1, 1)
    sns.heatmap(dice_df)
    ax.set_ylabel('')    
    ax.set_xlabel('')
    ax.set_yticklabels(ax.get_yticklabels(), rotation =0)
    ax.xaxis.tick_top()
    ax.set_xticklabels(ax.get_xticklabels(), rotation =0)
    ax.set_title('Dice coefficients/ similarity',y=1.08)   

 
# %%[Closest coordinate from set to another coordinate]

def closest_coordinate(coord, coord_set):
    """
    KDTree provides an index into a set of k-dimensional points which can be
    used to rapidly look up the nearest neighbors of any point.

    Parameters
    ----------
    coord : ndarray (size: 2)
        query coordinate
        
    coord_set : ndarray  (size: # coords x 2)
        all coordinates

    Returns
    -------
    close_coord : ndarray (size: 1 x 2)
        closest coordinate to the query point
        
    close_coord_idx : int (size: 1)
        index of closest coordinate

    """

    # create KD Tree from coordinates
    tree = spatial.KDTree(coord_set)
    
    # look up coordinate in tree
    tree_coord = tree.query([coord])
  
    # calculate the closest coordinate's index
    close_coord_idx = tree_coord[1]
    
    # get coordinate at that index
    close_coord = coord_set[close_coord_idx]
    
    
    return close_coord, close_coord_idx

 # %%[Centroids on map]
    
def map_hand_centroid(weights, region_key, hand_rp, **args):
    """
    centroids of the hand mesh on the map

    Parameters
    ----------
    weights: ndarray (size: # variables x # map units)
        map weights between each variable (Eg. afferent and map unit)
        
    region_key : str
        tag for a single region eg. 'D1d'
        
    hand_rp : dict
        hand regionprop data
        
    **args :
        ss_a : int
            sheet size in a dimension (default: sqrt map size)
        
        ss_b : int
            sheet size in b dimension (default: ss_a)

    Returns
    -------
    max_unit : int (size: 1)
        index of centroid map unit    
    
    coords : tuple
        centroid coordinate on the map

    """
    ss_a = args.get('ss_a', int(np.sqrt(np.size(weights,0))))
    ss_b = args.get('ss_b',ss_a)
    
    # get all the centroids
    # find closest afferent to the hand_pop centroid
    # get all afferents in the region
    aff_locs = hand_rp[region_key]['locs_hand']
    
    # get centroid
    centroid = hand_rp[region_key]['centroid']
    
    # find closest afferent to that centroid
    closest = closest_coordinate(centroid,aff_locs)[1]
    
    # find weight of that
    aff_index = hand_rp[region_key]['index'][closest]
    
    # find max unit
    max_unit = np.argmax(weights[aff_index,:])
    
    coords = np.unravel_index(int(max_unit),(ss_a,ss_b))    
    
    return max_unit, coords

# %%[centroid of the mesh]
 
def map_com_centroids(region_map):
    """
    calculates the centroid of a grid of values (the map)  

    Parameters
    ----------
    region_map: dict (size: ss_a x ss_b)
        reshaped map for a single region

    Returns
    -------
    coords : tuple
        centroid coordinate on the map

    """
    
    coords = ndimage.measurements.center_of_mass(region_map)
    
    # find nearest integer
    coords = np.round(coords)
    
    return coords

# %%[ Distance metric]
    
def distance_metric(**args):
    """
    calculate distance metric between:
        max_unit:
        COM (center of mass):
        hand_max_unit:

    Parameters
    ----------
    **args :
        method: str
            method for calculating the map centroids
            method for distance from 'max_unit', 'COM' and 'hand_max_unit'
            (default: 'max_unit')
        
        region_maps: dict (size: # groups)
            reshaped maps for each group/ region
            
        weights: ndarray (size: # variables x # map units)
            map weights between each variable (Eg. afferent and map unit)
        
        hand_rp : dict
            hand regionprop data
            
        region_key: list
            list of regions to calc distance
            
        ss_a : int
            sheet size in a dimension (default: 30)
        
        ss_b : int
            sheet size in b dimension (default: ss_a)        

    Returns
    -------
    max_unit : int (size: 1)
        index of centroid map unit    
    
    coords : tuple
        centroid coordinate on the map

    """
    
    method = args.get('method', 'max_unit') #max
    region_maps  = args.get('region_maps',None)
    weights = args.get('weights',None)
    hand_rp = args.get('hand_rp',None)
    region_key = args.get('region_key',None)
    ss_a = args.get('ss_a', 30)
    ss_b = args.get('ss_b',ss_a)
        
    if method == 'max_unit':
        coords = find_peak(region_maps[region_key])[0]
        max_unit = coords[0]*coords[1]
        
    elif method == 'com':
        coords = map_com_centroids(region_maps[region_key])
        max_unit = coords[0]*coords[1]
        
    elif method == 'hand_max_unit':
        max_unit, coords = map_hand_centroid(weights, region_key, hand_rp, ss_a=ss_a, ss_b=ss_b)
        
    return max_unit, coords

# %%[Centroids on map]

def map_centroids(**args):
    """
    calculate the centroids for all regions required

    Parameters
    ----------
    **args :
        method: str
            method for calculating the map centroids
            (default: hand_max_unit)
            
        region_maps: dict (size: # groups)
            reshaped maps for each group/ region
                           
        ss_a : int
            sheet size in a dimension (default: 30)
        
        ss_b : int
            sheet size in b dimension (default: ss_a) 
            
        group_name: list (size: no. of groups)
            groups of variables/ afferents in the map eg. regions ['D1', 'D2']
            
        hand_rp : dict
            hand regionprop data
            
        weights: ndarray (size: # variables x # map units)
            map weights between each variable (Eg. afferent and map unit)


    Returns
    -------
    max_unit : int (size: 1)
        index of centroid map unit    
    
    coords : tuple
        centroid coordinate on the map

    """
    # method: find unit closest to the centroid and use this point
    method = args.get('method', 'max_unit')
    region_maps  = args.get('region_maps',None)
    ss_a = args.get('ss_a', 30)
    ss_b = args.get('ss_b',ss_a)
    group_name = args.get('group_name',None)
    hand_rp = args.get('hand_rp',None)
    weights = args.get('weights',None)

    # max unit array
    max_unit = np.zeros((len(group_name),1))
    
    # max unit array
    coords = np.zeros((len(group_name),2))
    
    # get all the centroids
    for i in range(len(group_name)):
    
        max_unit[i],coords[i] = distance_metric(**args, region_key = group_name[i])  
    
    return max_unit, coords

# %%[distances]  

def map_hand_distances(weights, hand_pop, **args):
    """
    calculates distances between:
        -finger tips
        -medial phalanges
        -proximal phalanges

    Parameters
    ----------
    weights: ndarray (size: # variables x # map units)
        map weights between each variable (Eg. afferent and map unit)

    hand_pop: obj
          Touchsim afferentPopulation object
        
    **args :
        ss_a : int
            sheet size in a dimension (default: sqrt map size)
        
        ss_b : int
            sheet size in b dimension (default: ss_a)   
            
        method: str
            method for calculating the map centroids
            (default: hand_max_unit)
    Returns
    -------
    df_distance : pandas dataframe
        names of each region/ group pair and distance value

    """
    # get weight matrix and hand pop
    ss_a = args.get('ss_a', int(np.sqrt(np.size(weights,0))))
    ss_b = args.get('ss_b', ss_a)
    method = args.get('method','hand_max_unit')

    # calculate count
    hand_rp = hand_pop.hand_rp_sub
    
    # get keys
    group_name = sorted(list(hand_rp.keys()))
    
    # create region maps
    region_maps = region_weight_matrices(weights, hand_pop.hand_sub)[1]
    
    # order coordinates and hand tags alphabetically  
    _, coords_sort = map_centroids(method=method, group_name=group_name, hand_rp=hand_rp, weights=weights,
                                          ss_a=ss_a, ss_b=ss_b,region_maps=region_maps)
    
    sorted_ht= group_name
    
    #  calculate euclidean distance matrix
    dist = squareform(pdist(coords_sort, lambda u, v: np.sqrt(((u-v)**2).sum())))

    # create dataframe
    # make column names
    # rempmat group names and reshape
    col_1 = np.tile(sorted_ht,(len(sorted_ht),1))
    col_1 = np.reshape(np.transpose(col_1),[len(sorted_ht)**2])
    col_2 = np.tile(sorted_ht,len(sorted_ht))

    df_distance = pd.DataFrame(index=np.arange(0,len(col_1)),columns=['region_1','region_2','distance'])
    df_distance['region_1'] = col_1
    df_distance['region_2'] = col_2
    df_distance['distance'] = np.reshape(dist,[len(sorted_ht)**2]) 

    return df_distance

# %%[plots distances of the map- from centroids]    

def plot_distance_all(weights, hand_pop, **args):
    """
    calculates distances between:
        -finger tips
        -medial phalanges
        -proximal phalanges

    Parameters
    ----------
    weights: ndarray (size: # variables x # map units)
        map weights between each variable (Eg. afferent and map unit)
        
    hand_pop: obj 
            AfferentPopulation object, must have surface details attached.
        
    **args : 
        method: str
            method for calculating the map centroids
            (default: hand_max_unit)
            
    Returns
    -------
    None.

    """
    method = args.get('method','hand_max_unit')
        
    distance_df = map_hand_distances(weights, hand_pop, method=method)
    
    # Convert dataframe to pivot        
    distance_df = distance_df.pivot(index="region_1", columns="region_2", values="distance")
    
    # plot heatmap
    fig, ax = plt.subplots(1, 1,figsize=(12, 8))
    sns.heatmap(distance_df,annot=False,fmt='.1f',linewidths=.5)
    ax.set_ylabel('')    
    ax.set_xlabel('')
    ax.set_yticklabels(ax.get_yticklabels(), rotation =0)
    ax.xaxis.tick_top()
    ax.set_xticklabels(ax.get_xticklabels(), rotation =0)
    ax.set_title('Euclidean distance of centers',y=1.08)       
    
# %%[plots distances of digit regions on the map- from centroids]    
 
def plot_distance_phalanges(weights, hand_pop, **args):
    """
    calculates distances between:
        -finger tips (phalange = 'd')
        -medial phalanges (phalange = 'm')
        -proximal phalanges (phalange = 'p')

    Parameters
    ----------
    weights: ndarray (size: # variables x # map units)
        map weights between each variable (Eg. afferent and map unit)
        
    hand_pop: obj 
            AfferentPopulation object, must have surface details attached.
        
    **args : 
        method (str)
            method for calculating the map centroids
            (default: hand_max_unit)
            
    Returns
    -------
    None.

    """
    method = args.get('method', 'hand_max_unit')
    
    # get weight matrix and hand pop
    phalange = args.get('phalange', 'd')
        
    distance_df = map_hand_distances(weights, hand_pop, method=method)

    distance_df_cols = distance_df[(distance_df['region_1'].str.contains('D')) & (distance_df['region_1'].str.contains(phalange))]
    distance_df_cols = distance_df_cols[(distance_df_cols['region_2'].str.contains('D')) & (distance_df_cols['region_2'].str.contains(phalange))]

    # create pivot 
    # Convert dataframe to pivot        
    distance_df = distance_df_cols.pivot(index="region_1", columns="region_2", values="distance")
    
    # plot heatmap
    fig, ax = plt.subplots(1, 1,figsize=(12, 8))
    sns.heatmap(distance_df,annot=True,fmt='.1f',linewidths=.5)
    ax.set_ylabel('')    
    ax.set_xlabel('')
    ax.set_yticklabels(ax.get_yticklabels(), rotation =0)
    ax.xaxis.tick_top()
    ax.set_xticklabels(ax.get_xticklabels(), rotation =0)
    ax.set_title('Euclidean distance of centers',y=1.08)    


# %%[coords make dictionary]
    
def build_coord_dict(coords, group_name):
    """
    attaches the coordinates to the name of the corresponding region

    Parameters
    ----------
    coords : ndarray (size: no of groups x 2)
        coordinates
        
    group_name: list (size: no. of groups)
        groups of variables/ afferents in the map eg. regions ['D1', 'D2']

    Returns
    -------
    coord_dict : dict (size: no. of groups)
        keys: group names, value: coordinates

    """
    
    coord_dict = {}
    
    for i in range(len(group_name)):
        coord_dict[group_name[i]] = coords[i,:]
        
    return coord_dict

# %%[Hand mesh on map]

def map_mesh(weights, hand_pop, **args):
    """
    Adds mesh of the hand using the map centroids, method can be selected.

    Parameters
    ----------
    weights: ndarray (size: # variables x # map units)
        map weights between each variable (Eg. afferent and map unit)
        
    hand_pop: obj 
            AfferentPopulation object, must have surface details attached.
        
    **args :
        method: str
            method for calculating the map centroids
            (default: max_unit)
            
        labels: bool
            whether to display the names of each region code next to the point on the map.
        
        ss_a : int
            sheet size in a dimension (default: sqrt map size)
        
        ss_b : int
            sheet size in b dimension (default: ss_a)  

    Returns
    -------
    None.

    """
    method = args.get('method', 'max_unit')
    labels = args.get('labels',True)
    ss_a = args.get('ss_a', int(np.sqrt(np.size(weights,0))))
    ss_b = args.get('ss_b', ss_a)
    
    # create key lists for plotting
    group_name_sub = hand_pop.hand_sub['region_list']
    
    # calculate count
    hand_rp = hand_pop.hand_rp_sub
    
    # plot map
    fig, ax = view_map(weights, hand_pop.hand_main)

    # get map centroids
    region_maps = region_weight_matrices(weights, hand_pop.hand_sub)[1]
    peak_coords = map_centroids(method=method, group_name=group_name_sub, hand_rp=hand_rp, weights=weights, 
                                ss_a=ss_a, ss_b=ss_b ,region_maps=region_maps)[1]

    coord_dict = build_coord_dict(peak_coords,group_name_sub)
    
    ax.scatter(peak_coords[:,1],peak_coords[:,0]-0.25,30,color=[0.5,0.5,0.5])
    
    if labels:
        for i in range(len(group_name_sub)):
            ax.annotate(group_name_sub[i], (peak_coords[i,1], peak_coords[i,0]-0.5),color=[0.5,0.5,0.5],size=15)
        
    # get list for mesh
    connections = get_connections_list()
    
    # go through each list and plot
    for j in range(len(connections)):
        
        conn_list = connections[j]
        
        # loop through all combinations and plot for digit  
        for i in range(len(conn_list)):
            
            # get centroids for each in digit pair
            cx1 = coord_dict[conn_list[i][0]][1]
            cx2 = coord_dict[conn_list[i][1]][1]
            cy1 = coord_dict[conn_list[i][0]][0]-0.25
            cy2 = coord_dict[conn_list[i][1]][0]-0.25
            
            # create color map where palm connects up into the hand
            if conn_list[i][0][0] == 'D' and conn_list[i][1][0] == 'P':
                ax.plot([cx1,cx2],[cy1,cy2],10,color=[0.5,0.5,0.5]) #hand_rp[conn_list[i][1]][cmap])
            else:
                ax.plot([cx1,cx2],[cy1,cy2],10,color=[0.5,0.5,0.5]) #hand_rp[conn_list[i][0]][cmap])


# %%[plots numbers of unit on the map]    

def unit_code_map(weights, hand_data, **args):
    """
    Plots numbers on top of each unit in the map for easy reference

    Parameters
    ----------
    weights: ndarray (size: # variables x # map units)
        map weights between each variable (Eg. afferent and map unit)
    
    hand_data : Dict
        Hand region information
        
    **args :
        ss_a : int
            sheet size in a dimension (default: sqrt map size)
        
        ss_b : int
            sheet size in b dimension (default: ss_a)  

    Returns
    -------
    None.

    """
    ss_a = args.get('ss_a', int(np.sqrt(np.size(weights,0))))
    ss_b = args.get('ss_b', ss_a)
    
    group_name = hand_data['region_list']
    grouping_variable = hand_data['region_index']
    rgb_color_codes = hand_data['rgb_all']
    
    # input checks
    assert np.size(weights,1) == ss_a*ss_b, 'Number of map units and map sizes do not match, check ss_a and ss_b'
    assert len(grouping_variable) == np.size(weights,0), 'Number of variables in weights and grouping_variable do not match'
    assert len(group_name) == np.size(rgb_color_codes,0), 'Number of groups/ regions in group_name and RGB color_codes do not match'
    assert np.size(rgb_color_codes,1) == 3, 'each color in rgb_color_codes requires 3 r,g & b values'

    # plot map
    fig,ax = plot_map(weights, hand_data, save_name='')
    
    xlim = fig.axes[0].get_xlim()
    ylim = fig.axes[0].get_xlim()
    
    # plot a grid on top of the map
    x = np.linspace(xlim[0],xlim[1],ss_a+1)
    y = np.linspace(ylim[0],ylim[1],ss_b+1) 
    
    # text coords
    x_text, y_text = np.meshgrid(np.linspace(xlim[0],xlim[1]-1,ss_a),np.linspace(ylim[0]+0.7,ylim[1]-0.5,ss_b))
    x_text = x_text.flatten()
    y_text = y_text.flatten()
    
    # create array of numbers then convert to strings
    total_size = np.char.mod('%d', np.linspace(0,ss_b*ss_b-1,ss_a*ss_b))
    
    # add each of the number
    for i in range(len(y_text)):
        fig.axes[0].text(x_text[i],y_text[i],total_size[i],fontsize=8,color='tab:gray')
    
    hlines = np.column_stack(np.broadcast_arrays(x[0], y, x[-1], y))
    vlines = np.column_stack(np.broadcast_arrays(x, y[0], x, y[-1]))
    lines = np.concatenate([hlines, vlines]).reshape(-1, 2, 2)
    line_collection = LineCollection(lines, color='k', linewidths=1)
    fig.axes[0].add_collection(line_collection)

# # %%
# def total_map_activation(weights, **args):
#     """
#     Plots total weight across the map.

#     Parameters
#     ----------
#     weights: ndarray (size: # variables x # map units)
#         map weights between each variable (Eg. afferent and map unit)
        
#     ss_a : int
#         sheet size in a dimension (default: sqrt map_size)
    
#     ss_b : int
#         sheet size in b dimension (default: ss_a)  


#     Returns
#     -------
#     None.

#     """
#     ss_a = args.get('ss_a', int(np.sqrt(np.size(weights,0))))
#     ss_b = args.get('ss_b', ss_a)
    
#     plt.figure()
#     plt.imshow(np.sum(weights,0).reshape(ss_a,ss_b))
#     plt.colorbar()
#     plt.title('Weight sum across the map')

# # %%
# def plot_subregion_activation_map(weights, hand_pop, region_list_plot, ss_a, ss_b, plot_thresh=0, save_name=''):
#     """
#     Plots weight maps for specified regions.

#     Parameters
#     ----------
#     weights: ndarray (size: # variables x # map units)
#         map weights between each variable (Eg. afferent and map unit)
#     hand_pop: obj 
#             AfferentPopulation object, must have surface details attached.
#     region_list_plot : list
#         list of strings corresponding to regions to be plotted
#     ss_a : int
#         sheet size in a dimension (default: sqrt map size)
#     ss_b : int
#         sheet size in b dimension (default: ss_a)  
#     plot_thresh : float, optional
#         weight threshold for plotting. The default is 0.
#     save_name : str, optional
#         save name for plot. The default is ''.

#     Returns
#     -------
#     None.

#     """
    
#     region_list_sub = hand_pop.hand_sub['region_list']
    
#     region_index_sub = hand_pop.hand_sub['region_index']
    
#     region_index_plot = np.ones(len(hand_pop))*(len(region_list_plot))
    
#     for idx, j in enumerate(region_list_plot):
#        p = region_list_sub.index(j)
#        region_index_plot[region_index_sub[:,0] == p] = idx
    
#     _, _, overlap_weights = cortical_overlaps(weights, region_index_plot, region_list_plot, threshold = plot_thresh,
#                                                                                ss_a=ss_a, ss_b=ss_b)
    
#     view_map_all_regions(overlap_weights, region_list_plot, cmap_region='Blues_r')
#     plt.savefig(save_name)
    
    
