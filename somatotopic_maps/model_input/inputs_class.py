import numpy as np
import touchsim as ts
from touchsim.plotting import plot, figsave
from datetime import datetime
import os
from tqdm import tqdm

# %% class for hand inputs

class hand_input():
    """
    Map training inputs generated on the hand surface.
    
    Returns:
        Hand input object
    """
    
    def __init__(self, hand_pop, seed=0, p=None, tag=datetime.now().strftime("%Y%m%d_%H%M"), 
                 path='', stimuli_num=1000, visualize=False, save=False):
        """
        Parameters
        ----------
        hand_pop: obj 
                Touchsim AfferentPopulation object.
        seed : int, optional
            Seed for stimuli generators. The default is None.
        p : List, optional
            Stats for each area probability of contact. The default is None.
        tag : str, optional
            Savetag filename. The default is datetime.now().strftime("%Y%m%d_%H%M").
        path : str, optional
            Save path. The default is ''.
        stimuli_num : int, optional
            Number of stimuli. The default is 1000.
        visualize : bool, optional
            Plot each generated stimuli. The default is False.
        save : bool, optional
            Save data. The default is False.

        Returns
        -------
        None.

        """
        # Parameters for changing origin of plot hand (changes all coordinates into alternative system)
        self.seed = seed
        
        self.stimuli_num = stimuli_num
        
        # hand params
        self.input_data, self.input_stim_loc, self.pin_size, self.p, self.path_save = self.input_create(hand_pop.ts_hand, 
                                            seed=seed, p=p, tag=tag, path=path, stimuli_num=stimuli_num, visualize=visualize)
    
        self.aff_num = np.size(self.input_data,0)
        
        if save:
            self.save_inputs_separate(path=self.path_save)
            
            
    def input_create(self, ts_hand_pop, **args):
        '''
        Generates a matrix of responses of all afferents to a stimuli.
        matlab matrix output size is X*Y, where X is number of afferents and Y is
        number of stimuli presented.
        Folders must be set up in the save location before code is run (as defined
        by save path). The first level must be named'input_'+ name, within this
        there must be a
        folder named 'stimuli' for the stimuli information to be recorded.
    
        Parameters
        ----------
        hand_pop: obj 
                Touchsim AfferentPopulation object.
                
        **args : 
            seed: int 
                seed number (default: 1).
            p: list
                probability values of selecting each area (default: done by
                area size, uniform stimulation across the surface).
            tag: str 
                name of the file to be saved. Should be the name of the folder
                in save location.     
            path: str 
                path of save location (default: '', same folder currently in).
            stimuli_num: int 
                number of stimuli presented to the surface to generate
                responses.             
            visualise: bool
                Should be set to True if plots of the stimuli being
                placed in the region are required (default: False).
            stim_fun: function
                Stimulus generator function taking two arguments,
                1) loc, location of stimulus to be set by this script
                2) param, parameter matrix (stimuli_num X n) containing all parameters
                    that vary on each iteration.
    
        Returns
        -------
            files for inputs and stimuli locations and radius saved in save location.
        '''
        
        seed = args.pop('seed',1)
        
        # area probability
        p = args.pop('p', None)
        if p is None:
            # p is the probability list- % of time each area is contacted
            # get values from area property in surface. Calculate as a proportion out of 1.
            area_size = ts_hand_pop.surface.area;
            p = np.multiply(area_size,(1/np.sum(area_size)))
            # convert to list
            p = np.reshape(p, len(p))
            p = p.tolist()
    
        tag = args.pop('tag',datetime.now().strftime("%Y%m%d_%H%M"))
        path  = args.pop('path','')
        stimuli_num  = args.get('stimuli_num',1000)
        visualize  = args.pop('visualize',False)
    
        # set up stimulus function to call
        default_fun = lambda loc,param : ts.stim_ramp(loc=loc,pin_radius=param,
            amp=0.5,len=0.2,ramp_len=0.05)
        
        stim_fun = args.pop('stim_fun',default_fun)
    
        param = args.pop('stim_size',5+(10*np.random.rand(stimuli_num)))
    
        # create random seed
        np.random.seed(seed)
    
        # check hand_pop location
        a = ts_hand_pop.location
    
        # create empty array for responses
        x = np.empty((len(a),0))
    
        # create empty array for stimuli locations
        stim = np.empty((0,2))
    
        hand_surface = ts_hand_pop.surface
    
        # number of possible areas to be selected from
        select_num = hand_surface.num
    
        # create directory if it doesn't already exist
        
        save_path = f'{path}{tag}'
        
        if visualize:
            os.makedirs(f'{save_path}/inputs',exist_ok=True)
    
        # pre-allocate space for response and location matrices
        x = np.zeros((len(ts_hand_pop),stimuli_num))
        stim = np.zeros((stimuli_num,2))
    
        # for loop to create stimuli and responses
        for i in tqdm(range(stimuli_num), desc='Stim'):
    
            # selects one of the areas for the point to be located. P is the density, region sizes
            area_num = np.random.choice(select_num, 1, True, p=p)
            coord = hand_surface.sample_uniform(int(area_num[0]),num=1)
    
            # create stimulus at sample afferent
            s = stim_fun(loc=coord.tolist(),param=param[i])
    
            # add stimuli to stimuli matrix
            stim[i] = coord
    
            # calculate response to stimulus located at sampled afferent
            res = ts_hand_pop.response(s)
            r = res.rate()
    
            # add response to input response matrix
            x[:,i] = r.flatten()
    
            if visualize:
                fig = plot(hand_surface) * plot(s,spatial=True,surface=hand_surface) *\
                    plot(res,spatial=True)
                figsave(fig, f'{save_path}/inputs/inputs_{i}',
                    size=150,dpi=50)
        
        return x, stim, param, p, save_path
    
    def save_inputs_separate(self, path=''):
        """
        Save the input data.

        Parameters
        ----------
        path : str, optional
            Savepath. The default is ''.

        Returns
        -------
        None.

        """

        os.makedirs(path,exist_ok=True)
        
        # save inputs
        np.save(f'{path}/inputs_{self.seed}', self.input_data)
    
        # save stimulus location
        np.save(f'{path}/stim_loc_{self.seed}', self.input_stim_loc)
    
        # save stimulus parameters
        np.save(f'{path}/stim_pin_size_{self.seed}', self.pin_size)
    