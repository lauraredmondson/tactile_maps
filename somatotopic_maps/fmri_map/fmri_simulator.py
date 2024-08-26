'''fmri simulation and analysis
'''

#from somatotopic_maps.hand_model.create_locate_frame import create_regionprop
import numpy as np
import numpy.matlib
import touchsim as ts
import matplotlib.pyplot as plt

# %%[ 1) Create stimuli in blocks and trials, get responses using touchSIM]
'''
-Each trial has x number of blocks
-blocks are done as baseline and stimuli (mixed). This for fmri scanner, so we can ignore the purpose of blocks.
-Trials is the number of blocks that are carried out. We can use trial info to work out the total numbers of times each finger region is stimulated.
'''


def circle(radius, locs):
    angles = np.linspace(0, 2*np.pi, 50)
    return {'x': radius*np.sin(angles)+locs[0], 'y': radius*np.cos(angles)+locs[1], 'radius': radius}

# %%[fmri response to one stimuli]

def fmri_simulation_one_stimuli(**args):
    '''
    Simulates one digit tapping stimuli on the hand. Stimuli is placed centered
    on the hand centroid (some jittering of the stimulus is possible).
    Returns afferent responses

    Parameters
    ----------
    **args : 
        
        hand_pop: obj
              Touchsim afferentPopulation object
        
        region: str
            region where the stimuli is centered
        
        jitter: float
            jittering of the stimulus away from the centroid. Changes placement
            to random point within range of jitter (default: 0.5)
        
        stim_size: float
            radius of the stimuli (circle probe). (Default: 5)
        
        indent_amp: float
            Indentation of the pin (default: 0.5)
        
        ramp_len: float
            length of the stimuli ramp (default: 0.2)
            
        plot: bool
            whether to plot the stimuli (default: False)

    Returns
    -------
    stimuli_response : ndarray
        response of all the afferents to a tapping stimuli

    '''
    hand_pop = args.get('hand_pop')
    # region- list of full regions tags eg. D1d_t that will be contacted
    region = args.get('region')
    # jittering of the stimuli centroids
    jitter = args.get('jitter',0.5)
    # size (radius) of the circle probe.
    stim_size = args.get('stim_size',5)
    # indentation of the pin
    indent_amp = args.get('indent_amp',0.5)
    # ramp length
    ramp_len = args.get('ramp_len',0.05)
    # stim length
    stim_len = args.get('stim_len',0.2)
    # plot the stimuli
    plot = args.get('plot',False)
    
    # get information about the hand
    hand_rp = hand_pop.hand_rp_sub
    
    # get the region centroid
    centroid = hand_rp[region]['centroid']
         
    # create stimuli array, add jitter for each stimuli center if requested
    if jitter > 0:
        centroid[0]  = np.random.uniform(low=centroid[0]-jitter, high=centroid[0]+jitter)
        centroid[1]  = np.random.uniform(low=centroid[1]-jitter, high=centroid[1]+jitter)
    
    # create dictionary for all stimuli and response data
    stimuli_response={}
    
    # calculate stimuli, repsonse and rate to the centroid array
    stimuli_response['stimuli'] = ts.stim_ramp(loc=centroid,pin_radius=stim_size, amp=indent_amp,
                                                ramp_len=ramp_len, len=stim_len)
    
    # plot stimulus
    if plot:
        ts.plotting.plot(stimuli_response['stimuli'])
        
        fig, ax = plt.subplots()
        for i in range(len(hand_pop.surface.boundary)):
            plt.plot(hand_pop.surface.boundary[i][:,0],hand_pop.surface.boundary[i][:,1],color='gray')
        locs=hand_pop.surface.hand2pixel(stimuli_response['stimuli'].location[0])
        x =locs[0]
        y =locs[1]
        ax.scatter(x,y,5)
        convert_mm = hand_pop.surface.pxl_per_mm
        circle1 = plt.Circle((x,y), radius= stim_size*convert_mm,fill=False)
        ax.add_artist(circle1)
        plt.axis('equal')
    
    stimuli_response['responses'] = hand_pop.ts_hand.response(stimuli_response['stimuli'])
    stimuli_response['rate'] = np.squeeze(stimuli_response['responses'].rate()) 
    
    if plot:
        ts.plotting.plot(stimuli_response['responses'])
    
    return stimuli_response

# %%[runs the fmri experiment for one hand region]
def fmri_simulation_one_region(**args):
    '''
    Creates multiple stimuli in one region.
    Number of stimuli determined by number of trials.

    Parameters
    ----------
    **args : 
        hand_pop: obj
              Touchsim afferentPopulation object
            
        region: str
            region where the stimuli is centered
            
        trials: int
            Number of trials in each block (default: 50)
            
        jitter: float
            jittering of the stimulus away from the centroid. Changes placement
            to random point within range of jitter (default: 0.5)
        
        stim_size: float
            radius of the stimuli (circle probe). (Default: 5)
        
        indent_amp: float
            Indentation of the pin (default: 0.5)
        
        ramp_len: float
            length of the stimuli ramp (default: 0.2)
            
        plot: bool
            whether to plot the stimuli (default: False)

    Returns
    -------
    region_stimuli : dict (size: no. of trials)
        stimuli and responses for each stimulation.

    '''
    hand_pop = args.get('hand_pop')
    # regions- list of full regions tags eg. D1d_t that will be contacted
    region = args.get('region')
    # number of trials
    trials = args.get('trials',50)
    # jittering of the stimuli centroids
    jitter = args.get('jitter',0.5)
    # size (radius) of the circle probe.
    stim_size = args.get('stim_size',5)
    # indentation of the pin
    indent_amp = args.get('indent_amp',0.5)
    # ramp length
    ramp_len = args.get('ramp_len',0.05)
    # stim length
    stim_len = args.get('stim_len',0.2)
    # plot the stimuli
    plot = args.get('plot',False)
    
    # calculate total stimuli
    #total_stim_num = trials*blocks
        
    # create dictionary for all stimuli and response data
    region_stimuli = {}
    
    # for each stimuli and repetition
    for i in range(trials):
        region_stimuli['stim' + str(i)] = fmri_simulation_one_stimuli(hand_pop=hand_pop, region=region,
                    jitter=jitter, stim_size=stim_size, indent_amp=indent_amp, ramp_len=ramp_len, stim_len=stim_len, plot=plot)
        
    return region_stimuli

# %%[runs the fmri experiment for multiple hand regions]
   
def fmri_simulation_multi_regions(hand_pop, regions, **args):
    '''
    Creates stimuli and locations for each. Input options are typical to those 
    manipulated in fmri studies.

    Parameters
    ----------
    hand_pop: obj
              Touchsim afferentPopulation object
            
    regions: list
            list of regions to be stimulated
            
            
    **args :
        trials: int
            Number of trials in each block (default: 50)
        
        jitter: float
            jittering of the stimulus away from the centroid. Changes placement
            to random point within range of jitter (default: 0.5)
        
        stim_size: float
            radius of the stimuli (circle probe). (Default: 5)
        
        indent_amp: float
            Indentation of the pin (default: 0.5)
        
        ramp_len: float
            length of the stimuli ramp (default: 0.2)
            
        plot: bool
            whether to plot the stimuli (default: False)

    Returns
    -------
    stimuli_all : dict (size: no. of trials)
        stimuli and responses for each stimulation.


    '''
    # number of trials
    trials = args.get('trials',50)
    # jittering of the stimuli centroids
    jitter = args.get('jitter',0.5)
    # size (radius) of the circle probe.
    stim_size = args.get('stim_size',5)
    # indentation of the pin
    indent_amp = args.get('indent_amp',0.5)
    # ramp length
    ramp_len = args.get('ramp_len',0.05)
    # stim length
    stim_len = args.get('stim_len',0.2)
    # plot the stimuli
    plot = args.get('plot',False)

    
    # create dictionary for all data
    stimuli_all = {}    
        
    # calculate stimuli, repsonse and rate to the centroid array
    for i in range(len(regions)):
        stimuli_all[regions[i]] = fmri_simulation_one_region(hand_pop = hand_pop,region= regions[i],
                    trials=trials, jitter=jitter,stim_size=stim_size, stim_len=stim_len,
                    indent_amp=indent_amp,ramp_len=ramp_len,plot=plot)

    # return
    return stimuli_all

# %%[Get one region rates]
  
def fmri_one_region_rates(region_stimuli):
    '''
    Extract all rates for one region from one region's dictionary.

    Parameters
    ----------
    region_stimuli : dict (size: no. of trials)
        stimuli and responses for each stimulation.

    Returns
    -------
    rates : ndarray (size: no afferents x num stim)
        spike rates for each stimuli in one array

    '''
    
    rates = np.zeros((len(region_stimuli['stim0']['rate']),len(region_stimuli)))
    
    for i in range(len(region_stimuli)):
        rates[:,i] = region_stimuli['stim'+str(i)]['rate']
        
    return rates

# %%[Get one region rates]
    
def fmri_multi_region_rates(multi_region_stimuli, regions):
    '''
    Extract all rates for multi regions from multi region dictionary.

    Parameters
    ----------
    multi_region_stimuli: dict (size: no. of trials)
        stimuli and responses for each trial for each regions stimulation.
            
    regions: list
        list of regions

    Returns
    -------
    all_rates : dict (size: no. regions)
        contains ndarrays for each region with rates response for each stimuli
        on each afferent. narray size: (no.afferent x no. trials)

    '''

    all_rates = {}
    
    for i in range(len(regions)):
        all_rates[regions[i]] = fmri_one_region_rates(region_stimuli=multi_region_stimuli[regions[i]])
        
    return all_rates

