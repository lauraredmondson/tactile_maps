import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import scipy
from itertools import combinations
from scipy import stats

# %%[calculate the distances between responses using required distance metric]
    
def fmri_calc_distance(region_1, region_2, **args):
    '''
    Calculates the distances between two repsonse regions using required distance metric

    Parameters
    ----------
    region_1 : ndarray
        Responses for region 1
        
    region_2 : ndarray
        Responses for region 2
        
    **args : 
        distance_metric: str
            distance metric of euclidean, pearson, correlation (Default 'euclidean')

    Returns
    -------
    r_distance : ndarray
        array of distances between the points

    '''
    distance_metric = args.get('distance_metric','euclidean')
    
    if distance_metric == 'pearson':
        r_distance = 1-np.corrcoef(region_1,region_2)[0,1]
        
    elif distance_metric == 'spearman':
        r_distance,p  = stats.spearmanr(region_1.T,region_2.T) 
        r_distance = 1-r_distance
        
    elif distance_metric == 'euclidean':
        r_distance = (scipy.spatial.distance.euclidean(region_1,region_2))
    
    return r_distance

# %%[Calculate the distance between a pair of regions]
   
def fmri_RSA_analysis_one_pair(region_1, region_2, **args):
    '''
    Calculates the distance 'dissimilarity' between two repsonse regions.

    Parameters
    ----------
    region_1 : ndarray (size: ss_a x ss_b)
        Map responses for region 1
        
    region_2 : ndarray (size: ss_a x ss_b)
        Map responses for region 2
        
    **args : 
        distance_metric: str
            distance metric of euclidean, pearson, correlation (Default 'euclidean')
            
    Returns
    -------
    response_dissimilarity : float
        dissimilarity value of two maps

    '''

    distance_metric = args.get('distance_metric','euclidean')
    
    # reshape
    region_1 = np.reshape(region_1,-1)
    region_2 = np.reshape(region_2,-1)
    
    # correlate voxel responses of two regions
    response_dissimilarity = fmri_calc_distance(region_1,region_2,distance_metric=distance_metric)
    
    # return dissimilarity score
    return response_dissimilarity
    

  # %%[fMRI RSA analysis]
     
def fmri_RSA_analysis_all(average_maps, **args):
    '''
    Calculates complete RSA for all region pairs.

    Parameters
    ----------
    average_maps:  dict (size: no. regions)
        dictionary containing average response map over all trials for each region
    
    **args : 
        distance_metric: str
            distance metric of euclidean, pearson, correlation (Default 'euclidean')
            
        plot: bool
            whether to show the plot (default: False)

    Returns
    -------
    dissimilarity : ndarray (size no. region x no. regions)
        array of dissimilarity values

    '''
    distance_metric = args.get('distance_metric','euclidean')
    plot = args.get('plot',False)
        
    # get all region names
    region_names = list(average_maps.keys())
    
    r_key = {}
    
    for i in range(len(region_names)):
        r_key[region_names[i]] = i

    # use itertools combinations to obtain all pairs
    all_pairs = list(combinations(region_names,2))

    # add self value
    for i in range(len(region_names)):
        all_pairs.append((region_names[i],region_names[i]))
        
    # store dissimilarities
    dissimilarity = np.zeros((len(region_names),len(region_names)))
    
    # loop through pair list and find distance
    for i in range(len(all_pairs)):
        diss_value = fmri_RSA_analysis_one_pair(average_maps[all_pairs[i][0]],
                                                    average_maps[all_pairs[i][1]],distance_metric = distance_metric)
        dissimilarity[r_key[all_pairs[i][0]],r_key[all_pairs[i][1]]] = diss_value
        dissimilarity[r_key[all_pairs[i][1]],r_key[all_pairs[i][0]]] = diss_value
    
    # plot the dissimilarity matrix
    if plot:
        fmri_plot_RSA(dissimilarity, region_names)

    # return matrix
    return dissimilarity

    # %%[Dissimilarity RSA plot]
  
def fmri_plot_RSA(dissimilarity, region_names):
    '''
    Plots RSA for dissimilarity

    Parameters
    ----------
    dissimilarity : ndarray (size no. region x no. regions)
        array of dissimilarity values
        
    region_names : list
        Names of the regions corresponding to dissimilarity values

    Returns
    -------
    None.

    '''
    
    fig, ax = plt.subplots()
    im = ax.imshow(dissimilarity,cmap='Blues')
    ax.set_xticks(range(len(region_names)))
    ax.set_yticks(range(len(region_names)))
    ax.set_xticklabels(region_names)
    ax.set_yticklabels(region_names)
    fig.colorbar(im,ax = ax)
    ax.set_title('Dissimilarity')
    
    
        