# Analysing the inputs or results of the hand generator
import numpy as np
import matplotlib.pyplot as plt
from somatotopic_maps.map_plot_analysis.map_and_mesh_plots import sub_plot_panel
from collections import Counter

# %%[View image of the stimuli]
 
def plot_input_responses(inputs):
    '''
    Plots the afferent responses to each stimuli.

    Parameters
    ----------
    inputs : ndarray (size: afferents x afferent activation patterns)
        repsonses of the afferents to each stimuli

    Returns
    -------
    None.

    '''
    
    plt.figure()
    plt.imshow(inputs,aspect='auto')
    plt.xlabel('Response pattern')
    plt.ylabel('Afferent')
    plt.title('Input afferent responses')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel("Activation level") 

# # %%[View covariance of the responses]

# def plot_input_covariance(inputs):
#     '''
#     Plots the covariance of the afferent responses to all stimuli
#     A better plot to visualise this is 'plot_input_covariance_sorted'

#     Parameters
#     ----------
#     inputs : ndarray (size: afferents x afferent activation patterns)
#         repsonses of the afferents to each stimuli

#     Returns
#     -------
#     None.

#     '''
    
#     print('Use plot_input_covariance_sorted to see region labels')
    
#     input_cov = np.cov(inputs)
    
#     plt.figure()
#     plt.imshow(input_cov)
#     plt.xlabel('Afferent no.',fontsize=20)
#     plt.ylabel('Afferent no.',fontsize=20)
#     plt.title('Afferent response covariance',fontsize=20)
#     cbar = plt.colorbar()
#     cbar.ax.set_ylabel("Covariance",fontsize=15) 

# %%[View covariance of the responses]
# shows the covariances, sorted by hand region
def plot_input_covariance_sorted(inputs, hand_pop):
    '''
    Plots the covariance of the afferent responses to all stimuli.
    Sorted by regions.

    Parameters
    ----------
    inputs : ndarray (size: afferents x afferent activation patterns)
        repsonses of the afferents to each stimuli
        
    hand_pop : obj
        Touchsim hand population

    Returns
    -------
    None.

    '''
    region_list = hand_pop.hand_sub['region_list']
    aff_idx = hand_pop.hand_sub['region_index'].flatten()
    aff_sort = np.sort(aff_idx)
    aff_idx = np.argsort(aff_idx)

    aff_counts = np.asarray(list(Counter(aff_sort).values()))
    
    # location of last value of each region index set
    counts = np.cumsum(aff_counts)
    
    # sort
    inputs = inputs[aff_idx,:]
    
    # find covariance of inputs
    input_cov = np.cov(inputs)
    
    # number of afferents
    num_aff = len(input_cov)
    
    plt.figure()
    plt.imshow(input_cov,extent=[0, num_aff, 0, num_aff],origin='lower')
    plt.xlabel('Afferent region',fontsize=10)
    plt.ylabel('Afferent region',fontsize=10)
    plt.title('Afferent response covariance',fontsize=10)
    
    # calculate position of covariance points
    label_pos = np.append(0,counts)
    label_pos = label_pos[:20] + (aff_counts/2)
    
    plt.xticks(label_pos,region_list,rotation='vertical')
    plt.yticks(label_pos,region_list)
    
    x = range(num_aff)
    
    # add lines to separate the regions
    for i in range(len(counts)-1):
        
        y = np.tile(counts[i],num_aff)
        
        # plot horizontal line
        plt.plot(x, y, linewidth=1, color='r') 
        
        # plot vertical line
        plt.plot(y, x, linewidth=1, color='r')
    
    
    cbar = plt.colorbar()
    cbar.ax.set_ylabel("Covariance") 
    
# # %%[Stimuli in each location with stim locate]

# def stimuli_count_location(stim_locations, hand_pop):
#     '''
#     counts number of stimuli in each location. 
#     Does not include stimuli counts placed outside of the hand.
    
#     USE stimuli_count_peak_response for more accurate location.
    
#     Parameters
#     ----------
#     stim_locations : 
#         locations of each stimuli
        
#     hand_pop : obj
#         Touchsim hand population

#     Returns
#     -------
#     count_values : list
#         counts of stimuli in each region

#     '''

#     stim_locs = hand_pop.surface.locate(stim_locations)[1]
    
#     # remove -1:
#     remove_idx = np.where(stim_locs[stim_locs == -1])
#     stim_locs = np.delete(stim_locs,remove_idx)
    
#     # count number in each max_idx
#     count_stim = Counter(stim_locs)
#     count_values = list(count_stim.values())
#     count_keys = list(count_stim.keys())
    
#     loc_stim_response = {}
    
#     # get region name of key
#     for i in range(len(count_keys)):
#         d_key = hand_pop.surface.tags[count_keys[i]]
#         loc_stim_response[d_key] = count_values[i]
    
#     return count_values


# %%[Stimuli in each location based on peak response]

def stimuli_count_peak_response(inputs, hand_pop):
    '''
    Number of stimuli placed in each region based on where the peak response is.
    
    As some stimuli centers are placed outside of the hand, then their location 
    is not attached to a region. We estimate the appoximate location of the stimuli
    by calcualting which area has the largest response.
    
    This code picks up examples where stimuli is just outside the surface
    
    Parameters
    ----------
    inputs : ndarray (size: afferents x afferent activation patterns)
        repsonses of the afferents to each stimuli
        
    hand_pop : obj
        Touchsim hand population

    Returns
    -------
    loc_stim_response : dict
        region tags (keys) and corresponding number of stimuli placed in each (values)
        
    stim_locs_index : ndarray
        region index number of each of the stimuli
        
    location_tags : ndarray
        region tags of the placement of each stimuli

    '''

    # find the max for each input pattern
    max_idx = np.argmax(inputs,0)

    # locate afferents 
    locs_idx = hand_pop.ts_hand.surface.locate(hand_pop.ts_hand.location)[1]
    stim_locs_index = locs_idx[max_idx]
    
    location_tags = []
    
    # get afferent location tags for each input
    for i in range(len(stim_locs_index)):
        location_tags.append(hand_pop.ts_hand.surface.tags[stim_locs_index[i]])
    
    # count number in each max_idx
    count_stim = Counter(stim_locs_index)
    count_values = list(count_stim.values())
    count_keys = list(count_stim.keys())
    
    loc_stim_response = {}
    # get region name of key
    for i in range(len(count_keys)):
        d_key = hand_pop.ts_hand.surface.tags[count_keys[i]]
        loc_stim_response[d_key] = count_values[i]
    
    return loc_stim_response, stim_locs_index, location_tags

# In[stimuli as a percentage of each region]

def stimuli_count_percentage(count_values_dict):
    '''
    counts the number of stimuli in each region and returns the percentages
    in each region.

    Parameters
    ----------
    count_values_dict : dict
        region tags (keys) and corresponding number of stimuli placed in each (values)

    Returns
    -------
    count_percentage : ndarrray
        percentages of stimuli placed in each region

    '''
    
    count_values = np.fromiter(count_values_dict.values(),dtype='float')
    
    count_percentage = count_values/np.sum(count_values)*100
    
    return count_percentage


# %%[plot amount in each region]

def plot_input_counts(count_values_dict, hand_pop, **args):
    '''
    plots the number of stimuli placed in each regions

    Parameters
    ----------
    count_values_dict : dict
        region tags (keys) and corresponding number of stimuli placed in each (values)

    hand_pop : obj
        Touchsim hand population
        
    **args :
        type_count: str
            whether percentages for counts (default: counts)
            specify 'per' for percentages

    Returns
    -------
    None.

    '''
    type_count = args.pop('type_count','')

    labels = hand_pop.hand_sub['region_list']
    
    count_values_dict = dict(sorted(count_values_dict.items()))

    if type_count == 'per':
        count_values = stimuli_count_percentage(count_values_dict)
    else:
        count_values = np.fromiter(count_values_dict.values(),dtype='float')
            
    fig, ax = plt.subplots()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.bar(np.linspace(0,len(labels),len(labels)),count_values,color=hand_pop.hand_sub['rgb_all'])
    plt.xticks(np.linspace(0,len(labels),len(labels)),labels, rotation=90)
    plt.xlabel('Hand region')
    
    if type_count == 'per':
        plt.title('Percentage of total stimuli in each region')
        plt.ylabel('Percentage of stimuli')
    else:
        plt.title('Number of total stimuli in each region')
        plt.ylabel('Number of stimuli')
    
        
    

# # %%[Plot for multiple input sets]

# # sub plot panel
# def plot_input_counts_multiple(count_values_dict_multi, hand_pop, **args):
#     '''
#     plots inputs for different grasps ?? 

#     Parameters
#     ----------
#     count_values_dict_multi : dict
        
        
#     hand_pop : obj
#         Touchsim hand population
        
#     **args : 
#         type_count: str
#             whether percentages for counts (default: counts)
#             specify 'per' for percentages
            
#         names: list
#             names of the grasping types

#     Returns
#     -------
#     None.

#     '''
#     type_count = args.pop('type_count','')
#     names = args.pop('grasp_names',['uniform','weighted','precision','power'])
    
#     hand_gen = hand_pop.hand_as_generic(type_hand='sub')
        
#     # run subplot panelling- get back the two numbers for x and y panel size
#     panel = sub_plot_panel(len(names),space_h = 0.7, space_w = 0.7)
    
#     # create figure
#     fig = panel[0]
#     axes = panel[1]
    
#     fig.suptitle('Region maps')
    
#     # create map for each region and add
#     for i in range(len(names)):
        
#         name_dict = count_values_dict_multi[names[i]]
        
#         if type_count == 'per':
#             count_values = stimuli_count_percentage(name_dict)
#         else:
#             count_values = np.fromiter(name_dict.values(),dtype='float')

#         labels = list(name_dict.keys())
        
#         sort_labels_idx = np.argsort(labels)
#         sort_labels = np.sort(labels)
        
#         # sort 
#         count_values = count_values[sort_labels_idx]
        
#         cmap_bar = np.zeros((len(labels),3))
#         # create cmap array
#         for j in range(len(labels)):
#             idx_cmap = hand_gen[0].index(str(sort_labels[j]))
#             cmap_bar[j,:] = hand_gen[4][idx_cmap,:]
        
#         axes[i].bar(np.linspace(0,len(labels),len(labels)),count_values,color=cmap_bar)
#         axes[i].set_xticks(np.linspace(0,len(labels),len(labels)))
#         axes[i].set_xticklabels(sort_labels)
#         axes[i].set_xlabel('Hand region')
        
#         if type_count == 'per':
#             axes[i].set_ylabel('Percentage of stimuli in region')
#         else:
#             axes[i].set_ylabel('Number of stimuli in region')
        
#         # set title for figure
#         axes[i].set_title('Hand inputs for ' + names[i])
    
    

# # %%[Plot example response]

# def plot_stimuli_response(inputs, hand_pop):
#     '''
    
#     plots stimuli response on the hand

#     Parameters
#     ----------
#     inputs : ndarray (size: afferents x afferent activation patterns)
#         repsonses of the afferents to each stimuli
        
#     hand_pop : obj
#         Touchsim hand population

#     Returns
#     -------
#     None.

#     '''

#     t = np.linspace(0,50000,101) 
#     for i in t:
#         plt.figure()
#         plt.scatter(hand_pop.location[:,0],hand_pop.location[:,1],10,c=inputs[:,int(i)])
#         plt.savefig('images/' + str(int(i)))  

