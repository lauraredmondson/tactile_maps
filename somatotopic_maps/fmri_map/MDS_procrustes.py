
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from scipy.linalg import orthogonal_procrustes
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import matplotlib.patches as patches

# In[fMRI RSA analysis]
''' View the MD of RSA
'''    
def fmri_RSA_mds(dissimilarity, region_names, **args):
    '''
    Calculate the MDS of the RSA fitting

    Parameters
    ----------
    dissimilarity : ndarray (size: no. regions x no. regions)
        disssimilarity matrix
        
    region_names : list (size: no. regions)
        list of region names
        
    **args : 
        plot: bool
            whether to display the MDS plot (default: False)
            
        title: str
            title of the plot (default: '')

    Returns
    -------
    X_transformed : ndarray (size: no. regions x 2)
        MDS data with 2 extracted components (2D)

    '''
    num_iter_mds = args.pop('num_iter_mds',1)
    seed = args.pop('seed',0)
    plot = args.pop('plot',False)
    title = args.pop('title', '')
    
    # compute the MDS for 2D fitting
    embedding = MDS(n_components=2, dissimilarity="precomputed", metric=True, n_init=num_iter_mds, random_state=seed,normalized_stress=False)
    X_transformed = embedding.fit_transform(dissimilarity)
    
    # Compute the stress manually
    stress = np.sqrt(np.sum((dissimilarity - pairwise_distances(X_transformed, metric='euclidean'))**2)) / np.sqrt(np.sum(dissimilarity**2))
    print("Stress:", stress)

    #X_transformed.shape
    
    if plot:
        plt.figure()
        plt.scatter(X_transformed[:,0],X_transformed[:,1])
        
        for i, txt in enumerate(region_names):
            plt.annotate(txt, (X_transformed[i,0], X_transformed[i,1]))
        
        plt.title(title)
        
    return X_transformed


# %%[Mirror coords]
def basic_fit(coords_x, coords_y):
    '''
    Perform orthogonal procrustes algorithm fitting.

    Parameters
    ----------
    coords_x : ndarray (size: no. regions x 2)
        first set of coordinates of the MDS fitting for each region
        
    coords_y : ndarray (size: no. regions x 2)
        second set of coordinates of the MDS fitting for each region

    Returns
    -------
    coords_new : ndarray (size: no. regions x 2)
        Transformed coordinate fitting for the first set of coordinates.

    '''
    
    # run orthogonal procrustes between two set of coordinates
    R, sca = orthogonal_procrustes(coords_x,coords_y)
    
    # apply transformation on the coordinates
    coords_new = coords_x@R
    
    return coords_new


# %%

def plot_mds_single(mds_coords, hand_rp):
    """
    Plots the MDS for both map sims after procrustes fitting.

    Parameters
    ----------
    mds_coords : array
        MDS coords for map 1
    hand_rp : dict
        Hand pop information
    hand_tag : str, optional
        Hand region tag for legend. The default is 'd'.

    Returns
    -------
    None.

    """
    
    hand_regions = hand_rp['region_list'][:5]
    
    plt.figure()
    for i in range(len(mds_coords)):
        plt.scatter(mds_coords[i,0], mds_coords[i,1], label=hand_regions[i], color=hand_rp['rgb_all'][i,:], edgecolor='k')

    plt.legend()
    plt.axis('equal')




# %%

def plot_procrustes_fit(mds_coords, fit_coords, hand_rp, hand_tag='d'):
    """
    Plots the MDS for both map sims after procrustes fitting.

    Parameters
    ----------
    mds_coords : array
        MDS coords for map 1
    fit_coords : array
        Transformed MDS coords for map 2
    hand_rp : dict
        Hand pop information
    hand_tag : str, optional
        Hand region tag for legend. The default is 'd'.

    Returns
    -------
    None.

    """
    
    color_plot = hand_rp['rgb_all'][:5,:]
    hand_regions = hand_rp['region_list'][:5]
    
    fig, ax = plt.subplots()
    for i in range(len(mds_coords)):
        ax.scatter(mds_coords[i,0], mds_coords[i,1], label=hand_regions[i] + ' map 1', color=hand_rp['rgb_all'][i,:], edgecolor='k')

    for i in range(len(fit_coords)):
        ax.scatter(fit_coords[i,0],fit_coords[i,1], label=hand_regions[i] + ' map 2', color=hand_rp['rgb_all'][i,:], marker='s', edgecolor='k')
        
    # Define custom legend handles for the first part
    diamond_patches = [patches.Patch(color=color, label=f'D{i+1}{hand_tag}', linestyle='None') for i, color in enumerate(color_plot)]

    # Define custom legend handles for the second part
    map1_handle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10, label='Map 1')
    map2_handle = plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='black', markersize=10, label='Map 2')

    # Combine legend entries from the two parts
    handles1 = diamond_patches
    handles2 = [map1_handle, map2_handle]

    # Plot dummy data for second part
    ax.plot([], [], 'o', color='black', label='Map 1')
    ax.plot([], [], 's', color='black', label='Map 2')

    # Add the legend
    ax.legend(handles=handles1 + handles2, loc='upper left', bbox_to_anchor=(1, 1))
    
    ax_lims = np.max(np.abs([fit_coords,mds_coords]))*1.1
    ax.set_xlim([-ax_lims, ax_lims])
    ax.set_ylim([-ax_lims, ax_lims])
    ax.set_aspect('equal')





