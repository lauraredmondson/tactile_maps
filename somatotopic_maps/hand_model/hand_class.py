import numpy as np
import touchsim as ts
from collections import OrderedDict
import copy

# %% class for hand pop

class hand_data():

    """
    Hand data, with touchsim handpop.
    Adds info about hand regions and color codes/info for plotting
    
    Returns:
        hand_population object
    """
    
    def __init__(self, hand_tags=None, hand_density=None, image_filename=None,
                 affclass='SA1', density_multiplier=0.5): 
        """
        Initializes a hand input object.

        Parameters
        ----------
        hand_tags : dict, optional
            Hand tags for each region. The default is None.
        hand_density : dict, optional
            Densities for each region, afferents. The default is None.
        image_filename : str, optional
            Filename of the surface image. The default is None.
        affclass : str/list, optional
            Affclasses to be included in the touchsim hand. The default is 'SA1'.
        density_multiplier : float, optional
            Multiplier for overall afferent density. The default is 0.5.

        Returns
        -------
        None.

        """
        
        # hand params
        self.hand_tags = hand_tags
        self.hand_density = hand_density
        self.image_filename = image_filename
        
        self.affclass = affclass
        self.density_multiplier = density_multiplier

        # create the touchsim object
        self.create_touchsim_handpop()
        
        # Create region properties
        self.create_regionprop_main()
        self.create_regionprop_sub()
        
        self.hand_as_generic(type_hand='main')
        self.hand_as_generic(type_hand='sub')
        
        self.num_affs = len(self.ts_hand)

    def create_touchsim_handpop(self):
        '''
        Generates the touchsim handpop.

        Returns
        -------
        None.

        '''
        hand_orig = np.array([126.985990110355, 452.062407132244])
        hand_pxl_per_mm = 2.18388294387421
        hand_theta = -1.24458187646155
        
        # Create the surface object
        if self.image_filename is None:
            self.ts_hand = ts.affpop_surface(affclass=self.affclass, density_multiplier=self.density_multiplier)
            
        else:
            surface_toy = ts.Surface(filename=self.image_filename, orig=hand_orig,pxl_per_mm=hand_pxl_per_mm,
                    theta=hand_theta, density=self.hand_density, tags=self.hand_tags)
            
            self.ts_hand = ts.affpop_surface(surface=surface_toy, affclass=self.affclass, density_multiplier=self.density_multiplier)
    
    def create_regionprop_main(self):
        """
        Generates the region properties information (for each hand region) in dictionary.
        Includes tags for each region, afferent locations, color codes for plotting, centroids
        and area size.
        For main region eg. whole digits or palm.
        
        hand_rp : dict
            nested dictionary with the hand_pop split by region eg. main hand area or sub 
            hand area. Contains the following information:
                1) Area name/ region (determined by all codes in string)
                2) Locations on the hand in hand coordinates (determined by all codes in string)
                3) Locations on the hand in pixel coordinates (determined by all codes in string)
                4) Indexes of the locations in the main location array (determined by all codes in string)
                5) The colour value in the old system (determined by first codes in string)
                6) Centroid (determined by first codes in string)

        Returns
        -------
        None.

        """
        
        # original color codes, order is D1,D2,D3,D4,D5,P
        original_color = {'D1': [1,0,1],'D2': [1,0,0], 'D3': [1,1,0], 'D4': [0,1,0], 'D5': [0,1,1], 'Pw': [0,0,1], 'Pp': [0,0,1]}; 
        
        # create str tags for the regions eg. 'D1', 'D1d'
        region = []
        locs_tags_new = []
        locs_tags = self.ts_hand.surface.tags
        
        # creates the tags, if only main regions takes first two characters eg. 'D1'
        for i in range(len(locs_tags)):
            region_name = locs_tags[i]
            region.append(region_name[0:2])
            locs_tags_new.append(region_name[0:3]) 
        unique_regions = list(OrderedDict.fromkeys(region)) # get the unique region list
        
        # Create dictionary with the data.
        hand_rp = {} # create empty dict
        
        # use touchsim locate function to return the region location of each afferent
        [locs_aff,locs_idx] = self.ts_hand.surface.locate(self.ts_hand.location)
        
        # goes through each unique region, and adds data to all afferents with that
        for i in range(len(unique_regions)):       
            index_locs = np.where(np.asarray(region) == unique_regions[i])[0]

            index_array = []
            hand_rp[unique_regions[i]] = {} # add new entry to dictionary
            
            centroid_array = np.zeros([len(index_locs),2]) # create array to hold the region centroids
            area_array = np.zeros([len(index_locs)])
            
            # find index of all in index_locs
            for j in range(len(index_locs)):
                ii = np.where(np.asarray(locs_idx) == index_locs[j])[0]
                index_array.extend(ii) 
                # calculate centroid of the whole main region
                centroid_array[j] = self.ts_hand.surface.centers[index_locs[j]]
                area_array[j] = self.ts_hand.surface.area[index_locs[j]]
                        
            # add in the pixel locations of the afferents    
            hand_rp[unique_regions[i]]['locs_pixel'] = self.ts_hand.surface.hand2pixel(self.ts_hand.location[index_array])
            hand_rp[unique_regions[i]]['locs_hand'] = self.ts_hand.location[index_array] # add the hand locations
            hand_rp[unique_regions[i]]['index'] = np.array(index_array) # add the index in the main array of locations
            hand_rp[unique_regions[i]]['color'] = original_color[unique_regions[i]] # add the old color codes
            hand_rp[unique_regions[i]]['centroid'] = np.mean(centroid_array, axis=0) # add the centroid
            hand_rp[unique_regions[i]]['centroid_pixel'] = np.mean(self.ts_hand.surface.hand2pixel(centroid_array), axis=0) # add the rotated centroid
            hand_rp[unique_regions[i]]['area_size'] = np.sum(area_array, axis=0) # add the area_size
            hand_rp[unique_regions[i]]['area_size_per'] = np.sum(area_array, axis=0)/np.sum(self.ts_hand.surface.area)*100 # add the area_size
            
        self.hand_rp_main = hand_rp
    

    def create_regionprop_sub(self):
        """
        See regionprop main. For subregions of the hand.

        Returns
        -------
        None.

        """
        
        # original color codes, order is D1,D2,D3,D4,D5,P
        original_color = {'D1': [1,0,1],'D2': [1,0,0], 'D3': [1,1,0], 'D4': [0,1,0], 'D5': [0,1,1], 'Pw': [0,0,1], 'Pp': [0,0,1]}; 
        
        # create str tags for the regions eg. 'D1', 'D1d' sub tags required
        region = []
        locs_tags_new = []
        locs_tags = self.ts_hand.surface.tags
        
        # creates the tags, if only main regions takes first two characters eg. 'D1' if subregions, adds the sub tag eg 'D1d'
        for i in range(len(locs_tags)):
            region_name = locs_tags[i]
            region.append(region_name[0:3]) 
            locs_tags_new.append(region_name[0:3]) 
        unique_regions = list(OrderedDict.fromkeys(region))  # get the unique region list
         
        # Create dictionary with the data.
        hand_rp = {} # create empty dict
        
        # use touchsim locate function to return the region location of each afferent
        [locs_aff,locs_idx] = self.ts_hand.surface.locate(self.ts_hand.location)
        
        # goes through each unique region, and adds data to all afferents with that
        for i in range(len(unique_regions)):       
            hand_rp[unique_regions[i]] = {} # add new entry to dictionary
            generic_loc = unique_regions[i][0:2] # get generic location
            index_locs = np.where(np.asarray(region) == unique_regions[i])[0]
            ii = np.where(np.asarray(locs_idx) == index_locs[0]) # get index locations
            hand_rp[unique_regions[i]]['locs_pixel'] = self.ts_hand.surface.hand2pixel(self.ts_hand.location[ii]) # add the pixel locations
            hand_rp[unique_regions[i]]['locs_hand'] = self.ts_hand.location[ii] # add the hand locations
            hand_rp[unique_regions[i]]['index'] = np.array(np.squeeze(np.transpose(ii))) # add the index in the main array of locations
            hand_rp[unique_regions[i]]['color'] = original_color[generic_loc] # add the old color codes
            hand_rp[unique_regions[i]]['centroid'] = self.ts_hand.surface.centers[i] # add the centroid
            hand_rp[unique_regions[i]]['centroid_pixel'] = self.ts_hand.surface.hand2pixel(self.ts_hand.surface.centers[i]) # add the rotated centroid
            hand_rp[unique_regions[i]]['area_size'] = self.ts_hand.surface.area[i]  # add the area_size
            hand_rp[unique_regions[i]]['area_size_per'] =self.ts_hand.surface.area[i]/np.sum(self.ts_hand.surface.area)*100 # add the area_size

        self.hand_rp_sub = hand_rp
        
    def hand_as_generic(self, type_hand='main'):
        """
        returns the generic hand info from a hand population

        Parameters
        ----------
        type_hand: str
            whether main regions or sub regions wanted (default: 'main').
                
        Returns
        -------
        region_name : list (size no. regions)
            tags for each of the regions
            
        rgb_color_codes : ndarray (size no. unique colored regions x 3)
            color codes (RGB) for the colormaps
            this option returns unique colors in the array, typically both palm regions
            are blue, therefore it groups palm together into one color.
            Colormap is always returned ordered as D1,D2,D3,D4,D5,P
            
        afferent_color : ndarray (size no. afferents x 3)
            color code (RGB) of each afferent
            
        region_index : ndarray (size: # afferents x 1)
            region which each afferent belongs to- index of digit
            eg. [0] in group ['D1']

        rgb_color_codes_all : ndarray (size no. regions x 3)
             color codes (RGB) for the colormaps
             Returns all colors including duplicates (eg. both palm region are blue,
             returns both blue codes in array)
             Colormap is always returned ordered as D1,D2,D3,D4,D5,P

        """

        hand_dict = {}
        # get hand regionprop data
        if type_hand == 'main':
            hand_rp = self.hand_rp_main  
            self.hand_main = hand_dict
        else:
            hand_rp = self.hand_rp_sub
            self.hand_sub = hand_dict
            
        region_index = np.zeros([len(self.ts_hand.location),1])
            
        region_name = sorted(list(hand_rp.keys()))
        
        for i in range(len(hand_rp)):
            e = hand_rp[region_name[i]]['index']
            for j in range(len(e)):
                region_index[e[j]] = i
        
        hand_dict['region_list'] = region_name
        hand_dict['region_index'] = region_index
        hand_dict['afferent_color'], hand_dict['rgb_all'], hand_dict['rgb'] = self.calc_rgb_codes(hand_rp, region_name)

    def calc_rgb_codes(self, hand_rp, region_name):
        """
        Calculate the rgb codes for each afferent.

        Parameters
        ----------
        hand_rp : Dict
            Hand region properties
        region_name : List
            Regions of the hand

        Returns
        -------
        afferent_color : List
            Color codes for each afferent RGB.
        rgb_color_codes_all : List
            Color codes for all regions
        rgb_color_codes : List
            Color codes but palm merged (no separation between top and base of palm)

        """
        
        # create array, length is number of dimensions (here afferents), second is rgb code
        afferent_color = np.zeros([len(self.ts_hand.location),3])
       
        # create array for colormap
        rgb_color_codes = np.empty((len(region_name),3), int)
        
        # grouping variable of each afferent
        
        # get colour code of afferent locations
        for i in range(len(hand_rp)):
            
            # get indexes of all the afferents
            afferent_indexes = hand_rp[region_name[i]]['index']
                   
            # assign the colours at correct index to all those in first area
            for j in range(len(afferent_indexes)):
                
                afferent_index = afferent_indexes[j]
                afferent_color[afferent_index] = hand_rp[region_name[i]]['color']
                rgb_color_codes[i] = hand_rp[region_name[i]]['color']
        
        rgb_color_codes_all = copy.deepcopy(rgb_color_codes)    
        unique, index = np.unique(rgb_color_codes, axis=0, return_index=True)
        rgb_color_codes = unique[index.argsort()]
        
        return afferent_color, rgb_color_codes_all, rgb_color_codes
