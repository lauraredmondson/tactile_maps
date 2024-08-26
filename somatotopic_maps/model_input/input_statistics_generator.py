''' Generates input statistics based on different grasping types
    Feix grasp types included
    Data taken from Gonzalez
'''
 
import numpy as np
import pandas as pd
import touchsim as ts
from PIL import Image
import os.path

# file where the different hand grasps can be specified and touch stats can be returned
# optional inputs:
# Hand area eg. finger
# Grasping types (main eg. power, intermediate, precision, exploration)

# Returns statistics of contact for the required areas
# category axis is the Header group
# category list is the list of regions to be included
# list of required regions
# display image of graps- with numbers so can be selected

# %%[Display grasping types]    

def display_grasps():
    '''
    Displays image of grasping and exploratation
    Images taken from Gonzalez and Lederman. Can be used to create list of 
    grasping/ exploration codes

    Returns
    -------
    None.

    '''
    
    image = Image.open(os.path.dirname(__file__)  + '/Grasping_exploration_types.png')
    image.show()


def import_hand_codes():
    '''
    Imports the data from excel file with grasping numbers.

    Returns
    -------
    pandas dataframe
        hand statistics contact data as a pandas dataframe

    '''
    return pd.read_excel(os.path.dirname(__file__) + '/hand_codes.xlsx')
    
# %%[Get contact stats]
def get_grasp_stats(**args):
    '''
    Gets the grasping stats only from the dataframe.

    Parameters
    ----------
    **args : 
        cat_axis: str
            type of category axis (default: 'Category')
        
        cat_axis_list: list
            list of category axis values. (Default: ['E','M'])
    

    Returns
    -------
    df: pandas dataframe
        data frame with the selected data only

    '''
    
    cat_axis = args.get('cat_axis','Category')
    cat_axis_list = args.get('cat_axis_list',['E','M'])

    return_cols = args.get('return_cols',['Number','Contact raw'])

    df = import_hand_codes()

    idx = (df[cat_axis]==None).values # Boolean array, False
    for el in cat_axis_list:
        idx = np.logical_or(idx,(df[cat_axis]==el).values)

    return df[return_cols][idx]


# %%
# 
def get_contact_stats(**args):
    '''
    return the contact statistics requested.

    order of precedence: if grasps individual codes given this takes precedence eg.['K1', 'F29'], 
    otherwise cat/ cat_axis list(this can be Category or other)
    
    Parameters
    ----------
    **args : 
        regions: list
            list of region tags to be included in grasping stats (default: 'all')
            when default value uses all the regions.
            
        hand_pop: str
            Do not use this option
            * not tested yet with other hand shapes- maybe option not needed?
            (default: 'get' auto creates standard hand pop obj)
            

    Returns
    -------
    percentages: dict
        percentages for activations in each region (keys: hand tags, value: percentage active)

    '''
    
    # list of regions included
    regions = args.get('regions','all')   
    
    # cat axis list
    # cat_axis = args.get('cat_axis','Category')
    # cat_axis_list = args.get('cat_axis_list',['E','M'])
    hand_pop = args.get('hand_pop','get')
    
    # if no hand_pop object given, create standard hand shape
    # for area sizes
    if hand_pop == 'get':
        # create a new hand_pop surface
        hand_pop = ts.affpop_hand()
    
    # get region sizes from the hand_pop data
    region_sizes = hand_pop.surface.area
        
    # get region tags
    hand_strings = hand_pop.surface.tags
    
    # extract the first three chars from each hand string
    for i in range(len(hand_strings)):
        hand_strings[i] = hand_strings[i][:3]
        
    # create percentage code dictionary
    percentages = dict(zip(hand_strings,np.zeros((len(hand_strings)))))

    # get included indices
    df = import_hand_codes()
    values = get_grasp_stats(**args).index.values

    # select relevant regions- if we do not want all the regions, only 
    # select those requested
    if regions != 'all':     
        final_values = []
        
        # get list with all required fingers
        for i in range(len(values)):
            
            # go through the indexes and check the region code is included
            e = df['Codes'][values[i]]
            codes_keep = e.split(',')

            # check if region in included -> if yes, add index to final list
            for j in range(len(regions)):
                if regions[j] in codes_keep: 
                    final_values = np.append(final_values, values[i])
    
    else:
        regions = hand_strings
        final_values = values
                    
    # remove duplicates from final_values list
    final_values = np.unique(final_values)
    final_values = final_values[~np.isnan(final_values)]
    
     
    raw_values = np.zeros(len(final_values))
    
    for i in range(len(final_values)):
        raw_values[i] = df['Contact raw'][final_values[i]]
       
    code_list = np.zeros(len(final_values))
    
    total_region_size = 0
    
    for i in range(len(final_values)):
        # divide the raw value by the number of regions in the list that match
        
        # get the list of values for that grasp type
        code_list = df['Codes'][final_values[i]]
        code_list = code_list.split(',')
        
        # find the number of matching regions in the list of interest
        intersect = list(set.intersection(set(code_list),(set(regions))))

        # get the region sizes total for these areas
        
        # find the index and get region size
        
        # match the intersect value with the region list
        for j in range(len(intersect)):
            # get the region size value
            
            # find index of matching string
            index_hand = hand_strings.index(intersect[j])
            
            # add to total region size for the relevant values
            total_region_size += region_sizes[index_hand]
            
        # for each of the matching regions, add the value
        for j in range(len(intersect)):
            # get the region size and divide that by the raw_value
            
            # get percentage of total region size
            index_hand = hand_strings.index(intersect[j])
            # add to total region size for the relevant values

            area_value = region_sizes[index_hand]/total_region_size
            
            percentages[intersect[j]] += area_value*raw_values[i]
        
        total_region_size = 0
        
    # turn values into percentages    
    values = percentages.values()
    percentages_all = (list(values)/sum(values))*100
    
    # create percentage code dictionary
    percentages = dict(zip(hand_strings,percentages_all))
    
    return percentages


# %%[ exploratory/ manipulative]

def exploratory_manip_percentage(**args):
    '''
    give percentage of exploratory grasps, returns stats    

    Parameters
    ----------
    **args : 
        per_explor: int
            percentage of exploratory grasps

    Returns
    -------
    percentages: dict
        percentages for activations in each region (keys: hand tags, value: percentage active)

    '''
           
    per_explor = args.get('per_explor',50)
        
    exploratory_stats = get_contact_stats(cat_axis='Category',cat_axis_list=['E'])
    hand_strings = list(exploratory_stats)
    exploratory_stats = np.array(list(exploratory_stats.values()))
    
    manipulation_stats = np.array(list(get_contact_stats(cat_axis='Category',cat_axis_list=['M']).values()))
    
    percentages_all = exploratory_stats*(per_explor/100)+manipulation_stats*((100-per_explor)/100)
    
    # create percentage code dictionary
    percentages = dict(zip(hand_strings,percentages_all))
        
    return percentages


# %%[Sort statistics for input creator]
def sort_stats(hand_pop, prob):
    '''
    Gets a list of tags and probabilities to use with input_create

    Parameters
    ----------
    hand_pop: obj 
        Touchsim AfferentPopulation object.
    
    prob:
        
        
    Returns
    -------
    hand_tag : list
        hand region tags
        
    prob_list : list
        probabilities in each region

    '''
    # go through hand_pop and get the keys
    hand_tag = hand_pop.surface.tags

    prob_list = len(hand_tag)*[None]
    
    for i in range(len(hand_tag)):
        prob_list[i] = prob[hand_tag[i][0:3]]/100
    
    # Return tuple. First value is the list of order of probabilities. 
    # Second value is probability.
    return hand_tag, prob_list


           
