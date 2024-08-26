'''functions for different cortical maps and mesh'''
 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math

  # %%[other variable]
def plot_map(weights, hand_data, **args):
    """
    Assigns RGB color code to each corresponding afferent/variable in that 
    group.
    
    Plots the map using individual RGB codes.
    
    Parameters
    ----------
    weights: ndarray (size: # variables x # map units)
        map weights between each variable (Eg. afferent and map unit)
        
    hand_data : Dict
        Hand region information.
        
    **args :
        ss_a : int
            sheet size in a dimension (default: sqrt map size)
        
        ss_b : int
            sheet size in b dimension (default: ss_a)

        save_name : str
            save name of the plot (default: 'plot_map.png')
            
    Returns
    -------
    fig : matplotlib figure
        figure data
        
    ax : matplotlib axis
        axis data

    """
    # sheet sizes
    ss_a = args.pop('ss_a', int(np.sqrt(np.size(weights,0))))
    ss_b = args.pop('ss_b', ss_a)   
    save_name = args.pop('save_name','plot_map.png') # should be a string
    
    # check inputs
    assert np.size(weights,1) == ss_a*ss_b, 'Number of map units and map sizes do not match, check ss_a and ss_b'
    
    group_name = hand_data['region_list']
    group_index = hand_data['region_index']
    rgb_color_codes = hand_data['rgb_all']

    # create array, length is number of dimensions (here afferents), rows are rgb codes
    variable_colors = np.zeros([len(group_index),3])
    
    # get colour code of afferent locations
    for i in range(len(group_name)):
        
        # get indexes of all the variables in that group
        gv_indexes = np.where(group_index == i)
        
        # assign the colours at correct index to all those in each area
        for j in range(len(gv_indexes[0])):
            # add color code
            variable_colors[int(gv_indexes[0][j])] = rgb_color_codes[i]

    # run the map viewer
    fig, ax = view_map(weights, hand_data, ss_a=ss_a, ss_b=ss_b, save_name=save_name)
    
    return fig, ax
        
# %%[view the calculated map]   
def view_map(weights, hand_data, **args):
    """
    Visualises the map

    Parameters
    ----------
    weights: ndarray (size: # variables x # map units)
        map weights between each variable (Eg. afferent and map unit)
        
    hand_data : Dict
        Hand region information.
        
    **args : 
        ss_a: int
            sheet size in a dimension (default: sqrt map size)
        
        ss_b: int
            sheet size in b dimension (default: ss_a)
        
        cbar_label: 'str'
            title for the colorbar (default: 'Digit')
        
        save_name : str
            save name of the plot (default: 'plot_map.png')


    Returns
    -------
    None.

    """

    # sheet sizes
    ss_a = args.pop('ss_a', int(np.sqrt(np.size(weights,0))))
    ss_b = args.pop('ss_b', ss_a)   
    save_name = args.pop('save_name','plot_map.png') # should be a string
    cbar_label = args.pop('cbar_label', 'Digit')
    
    rgb_color_codes = hand_data['rgb_all']
    variable_colors = hand_data['afferent_color']
    group_name = hand_data['region_list']
    
    # check inputs
    assert np.size(weights,1) == ss_a*ss_b, 'Number of map units and map sizes do not match, check ss_a and ss_b'
    assert len(group_name) == np.size(rgb_color_codes,0), 'Number of groups/ regions in group_name and RGB color_codes do not match'
    assert np.size(rgb_color_codes,1) == 3, 'each color in rgb_color_codes requires 3 r,g & b values'

    # get average for red
    r = np.sum((variable_colors[:,0] * weights.T).T,0)/np.sum(weights,0)
    
    # get average for green
    g = np.sum((variable_colors[:,1] * weights.T).T,0)/np.sum(weights,0)
    
    # get average for blue
    b = np.sum((variable_colors[:,2] * weights.T).T,0)/np.sum(weights,0)
    
    # reshape sheets to cortical map size, [ss_a,ss_b] is reshape dimensions
    sheetr = np.reshape(r,[ss_a,ss_b])
    sheetg = np.reshape(g,[ss_a,ss_b])
    sheetb = np.reshape(b,[ss_a,ss_b])
    
    # create rgb map
    # stack rgb sheets   
    z = np.dstack((sheetr,sheetg,sheetb))
    
    # create figure
    fig, ax = plt.subplots()
    
    # show map
    ax.imshow(z)
    
    # set title for figure
    ax.set_title('Cortical map')

    # create colorbar
    my_cmap = mpl.colors.ListedColormap(rgb_color_codes, name='my_cmap')
    
    # add bounds for the colors
    bounds = np.linspace(0,1,len(rgb_color_codes)+1).tolist()

    # if map not square add colorbar horizontal
    if ss_a != ss_b:
        # add location of colorbar
        cbaxes = fig.add_axes([0.2, 0.25, 0.6, 0.05]) 
        orient_cbar = 'horizontal'
    else:
        orient_cbar = 'vertical'
        cbaxes = fig.add_axes([0.85, 0.1, 0.03, 0.8]) 
        
    dist = (1/len(group_name))/2
    # create custom colorbar
    cb = mpl.colorbar.ColorbarBase(ax = cbaxes, cmap=my_cmap,
                                    boundaries=bounds,
                                    ticks=np.array(bounds)[1:]-dist,
                                    orientation=orient_cbar)

    # add the tick labels
    if ss_a != ss_b:
        cb.ax.set_xticklabels(group_name)
    else:
        cb.ax.set_yticklabels(group_name) 
    
    # add the colorbar label
    cb.set_label(cbar_label)
    
    # turn off the axis box around the map
    ax.axis('off')

    #save as png
    plt.savefig(save_name)
    
    return fig, ax

# %%[view the calculated map]   

def view_map_one_region(weights_region, group_name, region_code, **args):
    """
    visualises the map for a single region. Gets the weights colours for the specified
    region and plots using a colormap provided or standard colormap

    Parameters
    ----------
    weights: ndarray (size: # variables in region x # map units)
        weight matrix for one region
        
    group_name: list (size: no. of groups)
        groups of variables/ afferents in the map eg. regions ['D1', 'D2']

    region_code : int
        index of corresponding region in group name to be plotted
        
    **args : 
        save_name : str
            save name of the plot (default: 'plot_map.png')
            
        cmap_region : str
            name of the matplotlib cmap to use (default: 'Blues_r')

    Returns
    -------
    None
    
    """
    
    # check code in weights_region
    # assert
    
    # save name for the map
    save_name = args.pop('save_name','plot_map.png') # should be a string
    cmap_region = args.pop('cmap_region','Blues_r')

    # create figure
    fig, ax = plt.subplots()
    
    # show map
    c = ax.imshow(weights_region[:,:,region_code],cmap=plt.get_cmap(cmap_region))
    
    # set title for figure
    ax.set_title('Cortical map for region '+ group_name[region_code])
    
    # turn off the axis box around the map
    ax.axis('off')
    
    fig.colorbar(c)

    #save as png
    plt.savefig(save_name)
    
    return fig

# %%[view the calculated map]   

def view_map_overlap(weights_region, group_name, region_code, **args):
    '''
    Show overlap between two regions on the map

    Parameters
    ----------
    weights_all : ndarray (size: ss_a x ss_b x len(group_name))
        map weights for each region, reshaped for map ploting
        
    group_name: list (size: no. of groups)
        groups of variables/ afferents in the map eg. regions ['D1', 'D2']
        
    region_code : list
        index of corresponding groups to be plotted
        
    **args :
        save_name : str
            save name of the plot (default: 'plot_map.png')
            
        rgb_color_codes: ndarray (size: # groups x 3)
            RGB color codes for each group/ region. 
            If not passed uses red and green

        ss_a: int
            sheet size in a dimension (default: 30)
        
        ss_b: int
            sheet size in b dimension (default: ss_a)

    Returns
    -------
    None

    '''
    # save name for the map
    save_name = args.pop('save_name','plot_map.png') # should be a string
    rgb_color_codes = args.pop('rgb_color_codes',None)
    ss_a = args.pop('ss_a', 30)
    ss_b = args.pop('ss_b',ss_a)   
    
    assert len(region_code) == 2, 'Two regions only should be specified in region_code'
    assert np.size(weights_region,0) == ss_a, 'First dimension size of weight_region should equal ss_a'
    assert np.size(weights_region,1) == ss_b, 'Second dimension size of weight_region should equal ss_b'
    assert np.size(weights_region,2) == len(group_name), 'Third dimension size of weight_region should equal len(group_name)'
    
    if rgb_color_codes is not None:
        cmap_final = np.zeros((4,3))
        cmap_final[0,:] = [1,1,1]
        cmap_final[1,:] = rgb_color_codes[region_code[0],:]
        cmap_final[2,:] = rgb_color_codes[region_code[1],:]
        cmap_final[3,:] = [0.5,0.5,0.5]
    else:
        cmap_final = np.array(([1,1,1],[1,0,0],[0,1,0],[0.5,0.5,0.5]))
    
    # map_size
    size = np.size(weights_region,1)*np.size(weights_region,0)
    
    # create overlap map
    overlap_map = np.zeros((size,3))
    
    # overlaps
    overlap = np.zeros((size,3))
    
    # create figure
    fig, ax = plt.subplots()
    
    # get the two matrices reshaped
    overlap[:,0] = np.squeeze(np.reshape(weights_region[:,:,region_code[0]],[size,1]))
    overlap[:,1] = np.squeeze(np.reshape(weights_region[:,:,region_code[1]],[size,1]))
    
    overlap[:,0][overlap[:,0] > 0] = 1
    overlap[:,1][overlap[:,1] > 0] = 2
    
    # find units where region 1 and 2 are active- add activation together
    overlap[:,2] = overlap[:,0]+overlap[:,1]
    
    # create colormap plot
    
    for i in range(len(overlap[:,2])):
        overlap_map[i,:] = cmap_final[int(overlap[i,2]),:]
    
    # create rgb map
    # stack rgb sheets   
    z = np.dstack((overlap_map[:,0],overlap_map[:,1],overlap_map[:,2]))
    
    z = np.reshape(z,[ss_a,ss_b,3])
    
    # show map
    ax.imshow(z)
    
    # set title for figure
    ax.set_title('Overlap map for '+ group_name[region_code[0]] + ' & ' + group_name[region_code[1]])
    
    # turn off the axis labels around the map
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.xticks([])
    plt.yticks([])
    
    # create colorbar
    my_cmap = mpl.colors.ListedColormap(cmap_final, name='my_cmap')
    
    # add bounds for the colors
    bounds = np.linspace(0,1,len(cmap_final)+1).tolist()
    
    # add location of colorbar
    # if map not square add colorbar horizontal
    if ss_a != ss_b:
        # add location of colorbar
        cbaxes = fig.add_axes([0.2, 0.25, 0.6, 0.05]) 
        orient_cbar = 'horizontal'
    else:
        orient_cbar = 'vertical'
        cbaxes = fig.add_axes([0.85, 0.1, 0.03, 0.8]) 
    
    dist = (1/len(cmap_final))/2
    
    # create custom colorbar
    cb = mpl.colorbar.ColorbarBase(ax = cbaxes, cmap=my_cmap,
                                    boundaries=bounds,
                                    ticks=np.array(bounds)[1:]-dist,
                                    orientation=orient_cbar)
    
    # add the tick labels
    if ss_a != ss_b:
        cb.ax.set_xticklabels(['Inactive\n both',group_name[region_code[0]],group_name[region_code[1]],'Overlap'])
    else:
        cb.ax.set_yticklabels(['Inactive\n both',group_name[region_code[0]],group_name[region_code[1]],'Overlap']) 
        
    #save as png
    plt.savefig(save_name)
    
    #return fig

# %%[view the calculated map]   
def view_map_all_regions(weights_regions, group_name, **args):
    """
    Figure with subplot panels showing map weights from each region 
    separately. 
    Colorbar scale equal over all maps.


    Parameters
    ----------
    weights_regions : ndarray
        returned from cortical overlaps function. Map for each region.
        
    group_name : list (size: no. of groups)
        groups of variables/ afferents in the map eg. regions ['D1', 'D2']
        Must match order in weights_regions

    **args :
        save_name : str
            save name of the plot (default: 'plot_map.png')
            
        cmap_region :  str
            name of the matplotlib cmap to use (default: 'Blues_r')
            
        region_codes : list (size: >=2)
            regions to be plotted if full list is not required
            (default: all regions in group_name)
            
    Returns
    -------
    None

    """
    
    # save name for the map
    save_name = args.pop('save_name','plot_map.png') # should be a string
    cmap_region = args.pop('cmap_region','Blues_r')
    region_codes = args.pop('region_codes',np.linspace(0,len(group_name)-1,len(group_name)).astype(int))
     
    assert len(region_codes) != 1, 'Use view_map_one_region to view single region map'
    
    # run subplot panelling- get back the two numbers for x and y panel size
    panel = sub_plot_panel(len(region_codes),space_h = 0.7, space_w = 0.7)
    
    # create figure
    fig = panel[0]
    axes = panel[1]
    
    fig.suptitle('Region maps')
    
    # create map for each region and add
    for i in range(len(region_codes)):
        # get map for first region
        sheet_final = weights_regions[:,:,region_codes[i]]
        
        # show map
        imshow_map = axes[i].imshow(sheet_final, cmap=plt.get_cmap(cmap_region), vmin=0, vmax=1)
        
        # set title for figure
        axes[i].set_title('Cortical map for ' + group_name[i])
        
        # turn off the axis box around the map
        axes[i].axis('off')
    
    # add colorbar for all
    cbar = fig.colorbar(imshow_map, ax=axes)
    cbar.set_label('Region weight to unit', rotation=90)
                   
    #save as png
    plt.savefig(save_name)
    
    return imshow_map


# %%[Sub plot panelling calculation]
    
def sub_plot_panel(plot_num, **args):
    """
    creates a subplot panel with the number of axes according to number of plots

    Parameters
    ----------
    plot_num : int
        number of axis panels
        
    **args :
        space_h : float
            space between the axes height (default: 1)
            
        space_w : float
            space between the axes width (default: 1)

    Returns
    -------
    fig1 : matplotlib figure
        empty figure panel data
        
    axs1 : matplotlib axis
        matplotlib axis data

    """
    space_h = args.get('space_h',1)
    space_w = args.get('space_w',1)
    
    # calculate grid size based on number of bases
    # check if divisible by square root, then work down until it fits
    if (np.sqrt(plot_num)).is_integer():
        grid_size = [np.sqrt(plot_num),np.sqrt(plot_num)]
        grid_size = [ int(x) for x in grid_size ]

    else:
        # find nearest square 
        square = math.ceil(np.sqrt(plot_num))
        grid_size = [square,square]
        if (grid_size[0]*grid_size[1])-plot_num > square:
            grid_size[1] = grid_size[1]-1
            grid_size = [ int(x) for x in grid_size]
            
    diff_size = (grid_size[0]*grid_size[1])-plot_num
    
    fig1, axs1 = plt.subplots(grid_size[0],grid_size[1], figsize=(15, 6), facecolor='w', edgecolor='k')
    fig1.subplots_adjust(hspace = space_h, wspace = space_w)
    
    # delete some axes if number of figures in grid is bigger than number of bases
    if diff_size > 0:
        # delete axis
        for i in range(diff_size):
            fig1.delaxes(axs1[grid_size[0]-1,(grid_size[1]-1)-i])

    axs1 = axs1.ravel()
    
    return fig1, axs1

# %%[plot mesh]

def COM_mesh(weights, locations):
    '''
    plots centre of mass mesh for any weight mesh and list of corresponding coordinates

    Parameters
    ----------
    weights: ndarray (size: # variables x # map units)
        map weights between each variable (Eg. afferent and map unit)
        
    locations : ndarray
        coordinate locations of the receptors

    Returns
    -------
    None.

    '''
    
    # places coordinate given a weighted contribution.
    masses = np.zeros((len(weights),3))
                 
    # add to array- X,Y and weights
    masses[:,:2] = locations
    
    weights_m = np.zeros((np.size(weights,1),2))

    # for each cortical unit calculate mesh centroid
    for i in range(np.size(weights,1)):
        # normalise W to equal 1
        weights_1 = weights[:,i]
        
        # add normalised weights as last column
        masses[:,2] = weights_1/np.sum(weights_1)
        
        # calculate centre of mass
        CM = np.average(masses[:,:2], axis=0, weights=masses[:,2])
        
        # add coordinate to weighted mean array
        weights_m[i] = np.transpose(CM)
    
    #w_m = hand_pop.surface.pixel2hand(w_m)

    size_w = int(np.sqrt(np.size(weights,1)))
    sheet_X = np.reshape(weights_m[:,0],[size_w,size_w])
    sheet_Y = np.reshape(weights_m[:,1],[size_w,size_w])

    fig = plt.figure()
    f = fig.add_subplot(111)
    f.plot(np.hstack((sheet_X,sheet_X.transpose())),np.hstack((sheet_Y,sheet_Y.transpose())),"-",color=(0,0,0),markersize=4)
    f.plot(sheet_X,sheet_Y,"o",color=(1,0,0),markersize=2)
    f.scatter(locations[:,0],locations[:,1],s=4)
    f.axis('scaled')
    f.axis('off')
    plt.show()
    
# %%[Plot coordinates on top of map]    
    
def plot_map_overlay_coord(coords, weights, hand_data):
    """
    plot the coordinates of the hand on top of the map.
    Typically the centroids of each region

    Parameters
    ----------
    coords : ndarray (size: len group list x 2)
        peak coords to be displayed over the map.
        
    weights: ndarray (size: # variables x # map units)
        map weights between each variable (Eg. afferent and map unit)

    hand_data : Dict
        Hand region information.

    Returns
    -------
    None.

    """
    group_name_sub = hand_data['region_list']
    
    # plot the map
    fig, ax = view_map(weights, hand_data)
    
    ax.scatter(coords[:,1],coords[:,0]-0.25,30,color=[0.5,0.5,0.5])
    
    for i in range(len(group_name_sub)):
        ax.annotate(group_name_sub[i], (coords[i,1], coords[i,0]-0.5),color=[0.5,0.5,0.5],size=15)


