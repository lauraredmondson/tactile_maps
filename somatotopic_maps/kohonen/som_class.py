import numpy as np
from somatotopic_maps.map_plot_analysis.map_and_mesh_plots import view_map
from tqdm import tqdm

class K_SOM():

    """
    Generates a weight array for the Kohonen SOM algorithm output.
    Output input array size is X*Y, where X is number of afferents and Y is
    number of cortical units. Each unit has a weight'strength of connection' to
    a cortical unit.
    
    Returns
    -------
    None.
        
    """
    
    def __init__(self, sheet_size_a = 30, sheet_size_b = 30,
                 seed = None, n_max= 1, n_min= 2,
                 n_decay= 0.005, l_max=0.05, l_min = 0.01, l_decay=0.001,
                 w=None):
        """
        Parameters
        ----------
        sheet_size_a : int, optional
            Size of map x. The default is 30.
        sheet_size_b : int, optional
            Size of map y. The default is 30.
        seed : int, optional
            Seed for map training. The default is None.
        n_max : float, optional
            Neighbourhood maximum size. The default is 1.
        n_min : float, optional
            Neighbourhood minimum size. The default is 2.
        n_decay : float, optional
            Neighbourhood decay rate. The default is 0.005.
        l_max : float, optional
           Maximum learning rate. The default is 0.05.
        l_min : float, optional
            Minimum learning rate. The default is 0.01.
        l_decay : float, optional
            Decay of learnign rate. The default is 0.001.
        w : ndarray, optional
            Weights predefined. The default is None.

        Returns
        -------
        None.

        """
        
        # sheet params
        self.sheet_size_a = sheet_size_a
        self.sheet_size_b = sheet_size_b
        self.total_sheet_size = sheet_size_a * sheet_size_b
        
        self.seed = seed
        
        # neighbourhood parameters
        self.n_max = int(np.max([sheet_size_a, sheet_size_b])*n_max) # initial neighbourhood radius, can be sheet_size_a/2
        self.n_min = n_min#int(np.max([sheet_size_a, sheet_size_b])*n_min) # final neighbourhood radius
        self.n_decay = n_decay # input_num/n_max

        # learning parameters
        self.l_max = l_max # initial learning rate
        self.l_min = l_min # final learning rate
        self.l_decay = l_decay # decay of learning rate
        
        self.w = w
    
    def build_map_units(self):
        """
        Build the units for the map for use with neighbourhood function

        Returns
        -------
        None.

        """
        # generate 2D sheet grid layout (cortical units)
        output_X, output_Y = np.meshgrid(
                            np.linspace(1,self.sheet_size_a,self.sheet_size_a),
                            np.linspace(1,self.sheet_size_b, self.sheet_size_b))

        return output_X.flatten(), output_Y.flatten()
    
    
    def build_weights(self):
        """
        Creates random weights for the initial map if none passed.

        Returns
        -------
        None.

        """
    
        # initial random weights
        weights = np.random.rand(self.num_receptor,self.cortical_sheet_size)
    
        # normalise weights
        self.w = np.divide(weights,sum(weights,2))


    def learning_rate(self, **args):
        '''
        calculates the learning rate for the kohonen SOM
    
        Parameters
        ----------
        **args : 
            l_max: float
                initial learning rate (default: 0.05)
                
            l_min: float
                final learning rate (default: 0.01)
                
            l_decay: float
                decay of learning rate (default: 0.01)
                
            input_num: int
                number of inputs to learn (default: 1000)
    
        Returns
        -------
        l_r : ndarray
            learning rate over time/ number of inputs
    
        '''
        l_max = args.pop('l_max', self.l_max) # initial learning rate
        l_min = args.pop('l_min', self.l_min) # final learning rate
        l_decay = args.pop('l_decay', self.l_decay) # decay of learning rate
        input_num = args.pop('input_num', 1000)
        
        # number of timesteps
        t_1 = np.linspace(0,1,input_num)
    
        # learning rate decay
        l_r = l_min+(l_max-l_min)*np.exp(-t_1/l_decay)
    
        return l_r
    
    
    def neighbourhood_size(self, **args): 
        '''
        calculates the neighbourhood size for the kohonen SOM
    
        Parameters
        ----------
        **args :
            n_max: int
                initial neighbourhood radius (default: self.n_max)
                
            n_min: int
                final neighbourhood radius (default: self.n_min)
                
            t_const: float
                decay constant for the neighbourhood size (default: self.n_decay)
                
            input_num: int
                number of inputs to learn (default: 1000)
        Returns
        -------
        n_s : ndarray
            neighbourhood sizes over time/ input patterns
    
        '''
        n_max = args.pop('n_max',self.n_max) # initial neighbourhood radius
        n_min = args.pop('n_min',self.n_min) # final neighbourhood radius
        t_const = args.pop('t_const',self.n_decay) # input_num/n_max
        input_num = args.pop('input_num', 1000)
        
        # number of timesteps
        t_1 = np.linspace(0,1,input_num)
        
        # neighbourhood size decay
        n_s = (n_min+(n_max-n_min)*np.exp(-t_1/t_const))
    
        return n_s
       
    def run_single_step(self, t, inputs, output_X, output_Y):
        """
        runs single training step

        Parameters
        ----------
        t : int
            current iteration
            
        inputs : ndarray
            inputs to train the model
            
        output_X : ndarray 
            unit indexes for x axis
            
        output_Y : ndarray 
            unit indexes for y axis

        Returns
        -------
        None.

        """
        # calculate dot product
        response = np.dot(self.w, inputs[:,t])
        
        if np.count_nonzero(response) > 0:
            # find winning cortical unit location
            r_max = np.argmax(response)
    
            # calculate neighbourhood extent around winning unit
            # select all units under or equal to a radius size threshold
            n = np.sqrt(((output_X-output_X[r_max])**2) + ((output_Y-output_Y[r_max])**2)) <= self.n_R2[t]
    
            # update weights using normalising Hebbian learning rule, learning rate
            # decreases over time
            dw = self.l_r[t]*(inputs[:,t]-self.w[n])
            
            self.w[n] += dw
              
            if self.weight_norm:
                row_sums = self.w(axis=1, keepdims=True)
                self.w /= row_sums
    
            # # calculate absolute change in weights
            self.w_change[t]= np.sum(np.abs(dw))/self.cortical_sheet_size
    
            #print(t)
    
            # add location of input
            if np.argmax(inputs[:,t]) >= 100:
                self.input_locs[t] = 1
            
            # add total amount of weight
            self.n_change[t,0] = np.sum(dw[:,:100])
            self.n_change[t,1] = np.sum(dw[:,100:])
        
        else:
            print('skipped input')


    def run_som(self, inputs, input_num=None, input_order='rand', update_plots=False, upd_data=[]):  
        """
        runs the initial SOM set up

        Parameters
        ----------
        inputs : ndarray
            inputs to train the model
        input_num : int, optional
            Number of input training patterns. The default is None (calcs based on input size).
        input_order : str, optional
            Order presentation of training patterns. The default is 'rand'.
        update_plots : bool, optional
            Plot update plots during training. The default is False.
        upd_data : List, optional
            Data for update plots. hand_rp and filename for save. The default is [].

        Returns
        -------
        None.

        """
        if input_num is None:
            input_num = np.size(inputs,1)
            
        assert input_num <= np.size(inputs,1)
            
        # create map shape/units
        output_X, output_Y = self.build_map_units()

        # number of timesteps
        t_1 = np.linspace(0,input_num,input_num)
        
        # neighbourhood size decay
        self.n_R2 = (self.n_min+(self.n_max-self.n_min)*np.exp(-t_1*self.n_decay))
    
        # learning rate decay
        self.l_r = self.l_min+(self.l_max-self.l_min)*np.exp(-t_1*self.l_decay)

        # vector of num_patterns length random integers, selected from number of responses
        if input_order == 'same':
            input_idx = np.arange(np.size(inputs,1))
        else:
           input_idx = np.random.choice(np.size(inputs,1), input_num, replace=False)
        
        # allocate array to identify which pattern is used (r0 or r1)
        self.input_locs = np.zeros((input_num))        
        
        # allocate arrays for weight change contribution provided from each region input
        self.n_change = np.zeros((input_num,2)) 
         
        # pre-allocate weight change vector
        self.w_change = np.zeros((input_num))
    
        # select the inputs to train
        inputs = inputs[:,input_idx[:input_num]]
        
        self.w = self.w.T

        # present one stimulus response per iteration to train the network
        #for t in range(input_num):
        for t in tqdm(range(input_num), desc='Iterations'):
            
            self.run_single_step(t, inputs, output_X, output_Y)
            
            # [Progress plots- plot map over time]
            if update_plots:
                if t in [1,100,300,500,600,1000,400,6000,8000]:
                        view_map(self.w.T, upd_data[0], ss_a=self.sheet_size_a, ss_b=self.sheet_size_b, save_name=upd_data[1])

        # set trained
        self._trained = True
        self.input_idx = input_idx
            

    def fit(self, inputs, input_num=None, input_order='rand', weight_norm=False, 
            update_plots=False, update_plot_data=[]):  # GROUP_NAME, COLOR_CODE, GV_INDEX
        """
        Runs fitting of the model

        Parameters
        ----------
        inputs : ndarray
            inputs to train the model
        input_num : int, optional
            Number of input training patterns. The default is None (calcs based on input size).
        input_order : str, optional
            Order presentation of training patterns. The default is 'rand'.
        weight_norm: bool, optional
            Whether to apply norm to weights during training
        update_plots : bool, optional
            Plot update plots during training. The default is False.
        upd_data : List, optional
            Data for update plots. hand_rp and filename for save. The default is [].

        Returns
        -------
        None

        """

        # set random seed
        if self.seed is not None:
            np.random.seed(self.seed)
            
        self.cortical_sheet_size = self.sheet_size_a*self.sheet_size_b
        self.num_receptor = np.size(inputs,0)
        
        # random weights if weights not already passed    
        if self.w is None:
            self.build_weights()
            
        # weight norm
        self.weight_norm = weight_norm
            
        # return weight array
        self.run_som(inputs, input_num, input_order, update_plots, upd_data= update_plot_data)