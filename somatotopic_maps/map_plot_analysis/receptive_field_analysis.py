# Receptive fields code
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import spatial
from scipy.spatial.distance import pdist, squareform
import math
from math import pi
from shapely.geometry import Polygon


# %%[circle boundary]
def circle_boundary(center, radius, num_points):
    """
    calcualate the boundary coordiantes of the circle

    Parameters
    ----------
    center : ndarray (size: 2)
        center coordiante of the circle
        
    radius : float
        radius
        
    num_points : int
        number of points to calculate for the circle boundary

    Returns
    -------
    points : ndarray (size: num points x 2)
        coordinates of the circle boundary

    """
    points = np.zeros((num_points,2))
    
    for x in range(num_points):
        points[x,:] = [center[0]+(math.cos(2*pi/num_points*x)*radius), center[1] + (math.sin(2*pi/num_points*x)*radius)] 
        
    return points    


# %%[Calculate for convex hull boundary]

def convex_hull_rf(coords,idx_keep):
    """
    calculate the convex hull boundary for the hand

    Parameters
    ----------
    coords : ndarray (size: num afferents x 2)
        coordinates of all the hand afferent locations
        
    idx_keep : ndarray (size: variable)
        indexes of all afferents over the threshold

    Returns
    -------
    area_raw : float
        retuns hand coverage on the skin. 
        
    rf_bound : ndarray (size: variable x 2)
        coordinate boundaries of the receptive field.
        Size depends on result of convex hull

    """
    
    # calculate convex hull
    coords_keep = coords[idx_keep,:]
    
    hull = spatial.ConvexHull(coords_keep)
    
    area_raw = hull.volume
    
    # get rf bounds
    rf_bound = coords_keep[hull.vertices,:]

    return area_raw, rf_bound

# %%[Calculate for circle boundary]

def circle_rf(coords,idx_keep):
    """
    calcualtes the circle receptive field

    Parameters
    ----------
    coords : ndarray (size: num afferents x 2)
        coordinates of all the hand afferent locations
        
    idx_keep : ndarray (size: variable)
        indexes of all afferents over the threshold

    Returns
    -------
    area_raw : float
        retuns hand coverage on the skin. 
        
    rf_bound : ndarray (size: 1000000 x 2)
        coordinate boundaries of the receptive field

    """
    
    # pdist of the values
    centre = np.array(([np.mean(coords[idx_keep[:],0])], [np.mean(coords[idx_keep[:],1])])).T
        
    r = squareform(pdist(np.vstack([centre,coords[idx_keep,:]])))
    
    # find furthest value (circle radius)
    radius = np.max(r[:,0])*1.05 # increase slightly to ensure point coverage due to shapely rounding
    
    # calculate area of the circle
    area_raw = np.pi*(radius**2)
    
    # get boundary of the circle
    rf_bound = circle_boundary(centre.T, radius, 100000)
    
    return area_raw, rf_bound
   
    
# %%[plots receptive fields on the hand using a threshold]    

def receptive_fields(weights, hand_pop, threshold, unit, **args):
    """
    calculates receptive fields and plots on the hand. 
    Plots extent of receptive field in 2D. 
    Returns size of the receptive field and centre.
    Units is the x,y coordinate of the map unit.
    Circle for time being, but should change to be 
    
    Receptive field coverage area calculations:
    'hull' - calculates the convex hull of the afferents within the receptive
    field (most accurate)
    'circle' - circle encompassing all the afferents (less accurate)

    Parameters
    ----------
    weights: ndarray (size: # variables x # map units)
        map weights between each variable (Eg. afferent and map unit)
        
    hand_pop: obj
          Touchsim afferentPopulation object

    threshold: float
        threshold of the receptive fields
        
    unit : int
        number of map unit to calculate the receptive field for
        
    **args : 
        plot: bool
            Plots the receptive fields (default: False)
            
        calc_type: str
            calculation type for the receptive field, options are 'hull' or
            'circle' (default:'hull')

    Returns
    -------
    rf_size : float
        size of the receptive field on the hand
        
    area_raw : float
        retuns hand coverage on the skin. 
        Can be different to rf_size if circle method used
        
    region_coverage : dict
        list of regions and size of area in that region covered
        
    region_size : ndarray (size: no. regions)
        sizes of each hand region
        
    dic_keys : dict (size: no. regions + 1)
        corrsponding keys for the rf and each region size

    """
    plot = args.get('plot',False)
    calc_type = args.get('calc_type','hull')
    
    # coords
    coords = hand_pop.ts_hand.surface.hand2pixel(hand_pop.ts_hand.location)
    
    unit_responses = weights[:,unit]
    
    # percentage out of 100
    unit_responses = unit_responses/(sum(unit_responses))*100
    
    # sort the responses and indexes high to low
    values = np.sort(unit_responses)
    index = np.argsort(unit_responses)
    
    values = values[::-1]
    index = index[::-1]
    
    # keep all values until threshold
    idx_keep = index[np.where(np.cumsum(values) <= threshold)[0]]
    
    # keep all values until that point
    values_keep = values[np.where(np.cumsum(values) <= threshold)[0]]
    
    # calculate receptive field type
    if calc_type == 'circle':
        area_raw, rf_bound = circle_rf(coords,idx_keep)
    
    else:
        area_raw, rf_bound = convex_hull_rf(coords,idx_keep)

    # dictionary to hold polygons
    dic ={}
    all_region_dic = {}
        
    # change to polygon
    dic['rf']=Polygon(list(rf_bound))    
    all_region_dic['rf']=Polygon(list(rf_bound))  
    
    # get boundaries of any hand surface shapes that have active afferents
    # find locations of the afferents 
    loc_data = hand_pop.ts_hand.surface.locate(hand_pop.ts_hand.location)
    locs = loc_data[1]
     
    rf_region_list = locs[idx_keep]
        
    # keep unique regions
    unique_regions = np.unique(rf_region_list)
    
    rf_size_coverage = np.zeros((len(unique_regions)))
    
    # change regions to polygons check overlap
    for i in range(len(unique_regions)):
        name = 'region_' + str(unique_regions[i])
        coords_reg = hand_pop.ts_hand.surface.boundary[unique_regions[i]]
        dic[name] = Polygon(coords_reg)
        # size
        p3=dic['rf'].intersection(dic[name])
        rf_size_coverage[i] = p3.area
    
    # recreate dictionary to include all regions        
    regions = np.unique(locs)
    
    region_size = np.zeros((len(regions)))
    
    for i in range(len(regions)):
        name = 'region_' + str(regions[i])
        coords_reg = hand_pop.ts_hand.surface.boundary[regions[i]]
        all_region_dic[name] = Polygon(coords_reg)
        region_size[i] = all_region_dic[name].area
                             
    dic_keys = list(all_region_dic.keys())
                
    if plot:
        plot_rf(coords, idx_keep, values_keep, all_region_dic)

    # returns hand coverage (skin)
    rf_size = np.sum(rf_size_coverage)
    
    # create info on size coverage of each region for the rf
    region_coverage = {}
    
    # add unique regions
    for i in range(len(unique_regions)):
        name = hand_pop.ts_hand.surface.tags[unique_regions[i]]
        # add coverage area in the region
        region_coverage[name] = rf_size_coverage[i]
    
    return rf_size, area_raw, region_coverage, region_size, dic_keys


# %%[Plot the receptive fields]
def plot_rf(coords, idx_keep, values_keep, all_region_dic):
    """
    plots the receptive fields on the hand

    Parameters
    ----------
    coords : ndarray (size: num afferents x 2)
        coordinates of all the hand afferent locations
        
    idx_keep : ndarray (size: variable)
        indexes of all afferents over the threshold
        
    values_keep : ndarray (size: len of ind_keep)
        values of all afferents over the threshold
        
    all_region_dic : dict (size: number of hand regions + 1)
        polygons of all the hand regions

    Returns
    -------
    None.

    """
    
    # plot the polygons to check
    fig, ax = plt.subplots(1)
    
    # scatter values
    ax.scatter(coords[idx_keep,0],coords[idx_keep,1],c=values_keep)

    dic_keys = list(all_region_dic.keys())

    for i in range(len(all_region_dic)):
        p = all_region_dic[dic_keys[i]]
        x, y = p.exterior.coords.xy
        points = np.array([x, y], np.int32).T
        if dic_keys[i] == 'rf':
            polygon_shape = mpl.patches.Polygon(points, linewidth=1, edgecolor='r', facecolor='none')
        else:
            polygon_shape = mpl.patches.Polygon(points, linewidth=1, edgecolor='k', facecolor='none')
        ax.add_patch(polygon_shape)
    
    plt.axis('off')
    plt.axis('scaled')
    plt.show()
    
    
# %%[plots receptive fields on the hand using a threshold]    
  
def rf_sizes(weights, hand_pop, **args):
    """
    calculates receptive fields and plots on the hand. 
    Plots extent of receptive field in 2D. 
    Returns size of the receptive field and centre.
    Units is the x,y coordinate of the map unit.
    Circle for time being, but should change to be convex hull

    Parameters
    ----------
    weights: ndarray (size: # variables x # map units)
        map weights between each variable (Eg. afferent and map unit)
        
    hand_pop: obj
          Touchsim afferentPopulation object
        
    **args : 
        threshold: float
            threshold of the receptive fields (default: 40)
        
        plot: bool
            Plots the receptive fields (default: False)

    Returns
    -------
    rf_sizes : ndarray (size: num map units)
        receptive field size for each map unit

    """
    threshold = args.get('threshold',40)
    plot = args.get('plot',False)
    
    # rf_size
    rf_sizes = np.zeros((np.size(weights,1)))
    
    # example to get area sizes         
    rf= receptive_fields(weights, hand_pop, threshold = threshold, unit=1)
    hand_sizes = rf[3][1:]
    
    # get hand area
    hand_area = np.sum(hand_sizes)

    #  calculate for each unit  
    for i in range(np.size(weights,1)):
        rf_sizes[i] = receptive_fields(weights, hand_pop, threshold = threshold, unit=i)[0]/hand_area*100

    
    if plot:
        plot_rf_sizes(rf_sizes)
        
    return rf_sizes  

# %%[Plot rf sizes]

def plot_rf_sizes(rf_sizes):
    """
    plots the rf sizes as a histogram

    Parameters
    ----------
    rf_sizes : ndarray (size: num map units)
        receptive field size for each map unit

    Returns
    -------
    None.

    """
    
    binwidth = 2
    
    # show histogram of sizes
    plt.hist(rf_sizes, bins=np.arange(0, 100 + binwidth, binwidth), facecolor='tab:blue')
    
    #plt.xlim([0,100])
    plt.xlabel('rf size as % of whole hand')
    plt.ylabel('Counts')
    plt.title('Range of rf sizes')
    plt.show()
    

# %%[plots region contribution to RF]    

def rf_region_contribution(weights, hand_pop, unit, **args):
    """
    calculates receptive fields and plots on the hand. 
    Plots extent of receptive field in 2D. 
    Returns size of the receptive field and centre.
    Units is the x,y coordinate of the map unit.
    Circle for time being, but should change to be convex hull

    Parameters
    ----------
    weights: ndarray (size: # variables x # map units)
        map weights between each variable (Eg. afferent and map unit)
        
    hand_pop: obj
          Touchsim afferentPopulation object
        
    unit : int
        number of map unit to calculate the receptive field for
        
    cmap : ndarray (size: number of map sub groups x 3)
        rgb color codes for each group
        
    **args :
        plot: bool
            Plots the receptive fields (default: False)

    Returns
    -------
    contribution : dict (size: no. of map subgroups)
        dictionary with keys: groups, and value: total map contribution from those
        groups to the map unit

    """
    plot = args.get('plot',False)
    
    cmap = hand_pop.hand_sub['rgb_all']
    
    unit_responses = weights[:,unit]
    
    # locate the afferents
    loc_data = hand_pop.ts_hand.surface.locate(hand_pop.ts_hand.location)
    locs = loc_data[1]
    
    # plots a bar chart with each region, and the percentage contribution from each
    contribution = {}
    
    # add unique regions
    for i in range(len(hand_pop.ts_hand.surface.tags)):
        name = hand_pop.ts_hand.surface.tags[i]
        # add coverage area in the region
        contribution[name] = np.mean(unit_responses[locs == i])

    contribution = {k: v / total * 100 for total in (sum(contribution.values()),) for k, v in contribution.items()}
    
    if plot:
        plot_rf_region_contribution(contribution,cmap,unit)
        
    return contribution
    
# %%[Receptive field regions plot]
 
def plot_rf_region_contribution(contribution, cmap, unit):
    """
    plots the receptive field contributions from each hand area/ group.

    Parameters
    ----------
    contribution : dict (size: no. of map subgroups)
        dictionary with keys: groups, and value: total map contribution from those
        groups to the map unit
        
    cmap : ndarray (size: number of map sub groups x 3)
        rgb color codes for each group
        
    unit : int
        number of map unit to calculate the receptive field for

    Returns
    -------
    None.

    """
    
    # bar chart of percentage contribution for each region
    y_pos = np.arange(len(contribution))
    
    counts = list(contribution.values())
    group_name = list(contribution.keys())
  
    # sort group name
    c_gn = sorted(zip(group_name,counts))
    c_gn = list(zip(*c_gn))
    group_name = list(c_gn[0])
    for i in range(len(group_name)):
        group_name[i] = group_name[i][:3]
  
    counts = c_gn[1]
    
    fig, ax = plt.subplots()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.bar(y_pos, counts, align='center', color=cmap, alpha=0.5)
    plt.xticks(y_pos, group_name,rotation=90)
    plt.ylabel('% of total unit weight')
    plt.xlabel('Region')
    plt.title('Map mean weight distribution-unit ' + str(unit))

    plt.show()
    