'''Hand analysis- specifically for use with hand shapes only eg. standard, primate, other
   Includes code to:
      Hand mesh
       - Calculates the distance along the hand mesh of any two given points
      Distances
       - Calculates the distance between fingers at each of the phalanges. 
             Returns array of distances for each phalange eg. tip, medial, proximal
      Area sizes
       - Plots the area size of each hand region
      Input analyais
       - Analysis of the inputs eg. spatial plot
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.spatial.distance import pdist, squareform


# %%
def get_hand_area_sizes(hand_rp):
    '''
    Area sizes for the hand from the hand region properties

    Parameters
    ----------
    hand_rp : Dict
        Hand region information.

    Returns
    -------
    sizes : List
        Area size of each region
    sizes_per : List
        Area size of each region (percentage)
    hand_area_tags : List
        List of regions corresponding to areas.

    '''

    hand_area_tags = sorted(hand_rp.keys()) # get hand_rp keys 
    sizes = np.zeros((len(hand_rp))) # create array for area size values
    sizes_per = np.zeros((len(hand_rp))) # create array for area size values

    for i in range(len(sizes)): # get counts
        sizes[i] = hand_rp[hand_area_tags[i]]['area_size']
        sizes_per[i] = hand_rp[hand_area_tags[i]]['area_size_per']

    return sizes, sizes_per, hand_area_tags
    
    
# %%[Plot hand area sizes]
def hand_area_sizes_plot(sizes_per, hand_rp, hand_area_tags, **args):
    '''
    Plot the hand area sizes as percentages

    Parameters
    ----------
    sizes_per : ndarray
        percentage sizes for each of the regions
        
    hand_rp : Dict
        Hand region information.
        
    hand_area_tags : list
        corresponding region tags for each of the items in sizes_per
        
    **args : 
        type_hand : str
            Whether the tag type is for main or subregions (default: 'main')

    Returns
    -------
    None.

    '''
    type_hand = args.get('type_hand','main')

    # create cmap 
    cmap = np.zeros((len(hand_rp),3))   
    for i in range(len(sizes_per)):
        cmap[i] = hand_rp[hand_area_tags[i]]['color']

    fig, ax = plt.subplots()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    y_pos = np.arange(len(hand_area_tags))

    plt.bar(y_pos, sizes_per, align='center', color=cmap, alpha=0.5)
    
    plt.xticks(y_pos, hand_area_tags)
    
    # change xticks to be rotated if plotting sub regions
    if type_hand == 'sub':
        ax.set_xticklabels(ax.get_xticklabels(), rotation =90)
        
    plt.ylabel('% of hand size')
    plt.xlabel('Hand region')
    plt.title('Hand region sizes')

    plt.show()
    
 
# %%[Plot hand area sizes]
def get_centroids(hand_rp):
    '''
    Plot centroids on the hand image.

    Parameters
    ----------
    hand_rp : Dict
        Hand region information.

    Returns
    -------
    centroids : ndarray
        Array of centroids of each hand region
        
    centroid_keys: list
        Names of the regions corresponding to the centroids

    '''
    # get hand_rp keys     
    centroid_keys = sorted(hand_rp.keys())
   
    # create array for area size values
    centroids = np.zeros((len(hand_rp),2))
    
    # get counts
    for i in range(len(centroids)):
        centroids[i,:] = hand_rp[centroid_keys[i]]['centroid_pixel']

    return centroids, centroid_keys


# %%[Plot hand area sizes]
def plot_centroids(centroids, hand_pop):
    '''
    Plot centroids on the hand image.
    If no arguments passed, creates Touchsim afferentPopulation for plotting.
    
    Parameters
    ----------
    centroids : ndarray
        Array of centroids of each hand region
        
    hand_pop: obj
        Touchsim afferentPopulation object, must have surface details attached

    Returns
    -------
    None.

    '''

    # plot hand_pop surface
    hand_boundaries = hand_pop.ts_hand.surface.boundary
    for i in range(len(hand_boundaries)):
        plt.plot(hand_boundaries[i][:,0],hand_boundaries[i][:,1],c='grey')
        
    plt.scatter(centroids[:,0], centroids[:,1])
    
    plt.title('Hand area centroids')
    plt.axis('equal')
    plt.axis('off')


# %%[calc distances]  
    
def calc_hand_distances(hand_rp):
    '''
    Calculates distances between the centroids on the whole hand  

    Parameters
    ----------
    hand_rp : Dict
        Hand region information.

    Returns
    -------
    hand_distances : Pandas dataframe
        all distances between centroids in each region

    '''
    # order coordinates and hand tags alphabetically  

    # get hand_rp keys     
    sorted_hand_tags = sorted(hand_rp.keys())
   
    # create array for area size values
    coords_sort = np.zeros((len(hand_rp),2))
    
    # get coordinates of centroids for each region
    for i in range(len(coords_sort)):
        coords_sort[i,:] = hand_rp[sorted_hand_tags[i]]['centroid']
    
    #  calculate euclidean distance matrix of the coordinates
    dist = squareform(pdist(coords_sort, lambda u, v: np.sqrt(((u-v)**2).sum())))

    # create dataframe with hand distance data
    col_1 = np.tile(sorted_hand_tags,(len(sorted_hand_tags),1))
    col_1 = np.reshape(np.transpose(col_1),[len(sorted_hand_tags)**2])
    col_2 = np.tile(sorted_hand_tags,len(sorted_hand_tags))

    hand_distances = pd.DataFrame(index=np.arange(0,len(col_1)),columns=['region_1','region_2','distance'])
    hand_distances['region_1'] = col_1
    hand_distances['region_2'] = col_2
    hand_distances['distance'] = np.reshape(dist,[len(sorted_hand_tags)**2]) 

    return hand_distances
    
    
# %%[Plot euclidean distances]  
  
def plot_hand_distances(hand_distances):
    '''
    Plots heatmap of distances for all hand regions

    Parameters
    ----------
    hand_distances : Pandas dataframe
        all distances between centroids in each region

    Returns
    -------
    None.

    '''

    # Convert dataframe to pivot        
    hand_distances_pivot = hand_distances.pivot(index="region_1", columns="region_2", values="distance")
    
    # plot heatmap
    fig, ax = plt.subplots(1, 1,figsize=(12, 8))
    sns.heatmap(hand_distances_pivot,annot=True,fmt='.1f',linewidths=.5)
    ax.set_ylabel('')    
    ax.set_xlabel('')
    ax.set_yticklabels(ax.get_yticklabels(), rotation =0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation =90)
    plt.title('Euclidean distance of centroids')    
        
    
# %%[Plot euclidean distances]  
  
def hand_distances_pair(hand_distances, region_1, region_2):     
    '''
    Single distance calculation distance between a pair of regions

    Parameters
    ----------
    hand_distances : Pandas dataframe
        all distances between centroids in each region
        
    region_1 : str
        tag of the first region name eg. 'D1' or 'D1d'
        
    region_2 : str
        tag of the second region name eg. 'D2' or 'D2d'

    Returns
    -------
    distance : float
        euclidean distance between the two regions

    '''
    
    assert len(region_1) == len(hand_distances.region_1[0]), "region tags are different type to hand distance tags (main/ sub)"
    assert len(region_2) == len(hand_distances.region_2[0]), "region tags are different type to hand distance tags (main/ sub)"
    
    # add to list
    regions = [region_1,region_2]
    
    # sort
    regions = sorted(regions)
    
    # change to correct order
    region_1 = regions[0]
    region_2 = regions[1]
    
    idx = np.where((hand_distances.region_1 == region_1) & (hand_distances.region_2 == region_2))
    distance = hand_distances.distance[idx[0][0]]

    return distance
        
# %%[afferent area counts plot]

def plot_aff_counts(sort_keys, per_aff_counts, hand_rp, **args):
    """
    plots the afferent percentages in each region of the hand population

    Parameters
    ----------
    sort_keys : list
        corresponding region tags for the aff count data
        
    per_aff_counts : ndarray
        % of afferents in each hand region
        
    hand_rp : Dict
        Hand region information.
        
    **args : 
        type_hand: str
            whether main or sub regions (default: 'main')
             
    Returns
    -------
    None.

    """
    
    type_hand = args.get('type_hand','main')
    
    fig, ax = plt.subplots()
        
    # set axis label
    y_axis_label = 'Percentage of afferents'
    y_pos = np.arange(len(sort_keys))

    # create cmap array if none given
    cmap = np.zeros((len(hand_rp),3))   
    for i in range(len(per_aff_counts)):
        cmap[i] = hand_rp[sort_keys[i]]['color']
        
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.bar(y_pos, per_aff_counts, align='center', color=cmap, alpha=0.5)
    plt.xticks(y_pos, sort_keys)
    plt.ylabel(y_axis_label)
    plt.title('% of total afferents in each region')
    
    if type_hand == 'sub':
        ax.set_xticklabels(ax.get_xticklabels(), rotation =90)

    plt.show()
    
# %%[afferent area counts]

def aff_counts(hand_pop, **args):
    '''
    counts the number of afferents in each region from the hand population
    plots if required

    Parameters
    ----------
    **args : 
        hand_pop: obj
            Touchsim afferentPopulation object
        
        type_hand: str
            whether main or sub regions (default: 'main')
             
        cmap: ndarray (size no. regions x 3)
            array of RGB codes (default: None (plots with old color scheme))
            
        plot: bool
            Plots the afferent percentages (default: False)
        

    Returns
    -------
    aff_count : ndarray
        no. of afferents in each hand region
        
    per_aff_counts : ndarray
        % of afferents in each hand region
        
    sort_keys : list
        corresponding region tags for the aff count data

    '''
    type_hand = args.get('type_hand','main')
    plot = args.get('plot',False)
    
    if type_hand == 'main':
        hand_rp = hand_pop.hand_rp_main
    else:
        hand_rp = hand_pop.hand_rp_sub
    
    # get hand_rp keys     
    sort_keys = sorted(hand_rp.keys())
   
    # create array for area size values
    aff_count = np.zeros((len(hand_rp)))

    # get counts
    for i in range(len(aff_count)):
        aff_count[i] = len(hand_rp[sort_keys[i]]['index'])

    # calculate percentage of areas- plot as percentage, update axis labels
    per_aff_counts = aff_count/np.sum(aff_count)*100

    if plot:
        plot_aff_counts(sort_keys, per_aff_counts, hand_rp, type_hand= type_hand)

    return aff_count, per_aff_counts, sort_keys
 
 