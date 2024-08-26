# Hand mesh
from .hand_analysis import calc_hand_distances, hand_distances_pair
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

 # %%[Hand mesh]      

def hand_mesh(hand_pop):
    '''
    Plots the hand mesh on the hand image

    Parameters
    ----------
    hand_pop: obj
          Touchsim afferentPopulation object

    Returns
    -------
    None.

    '''
    
    # boundary
    hand_surface = hand_pop.ts_hand.surface
    
    plt.scatter(hand_surface._centers[:,0],hand_surface._centers[:,1]) 
    
    # get the centroids and colors
    hand_rp = hand_pop.hand_rp_sub   
    
    # get connections
    connections = get_connections_list()

    # plot hand_pop surface
    for i in range(len(hand_surface.boundary)):
        plt.plot(hand_surface.boundary[i][:,0],hand_surface.boundary[i][:,1],color='grey')
    
    # go through each list and plot
    for j in range(len(connections)):
        
        conn_list = connections[j]
        
        # loop through all combinations and plot for digit  
        for i in range(len(conn_list)):
            
            # get centroids for each in digit pair
            cx1 = hand_rp[conn_list[i][0]]['centroid_pixel'][0]
            cx2 = hand_rp[conn_list[i][1]]['centroid_pixel'][0]
            cy1 = hand_rp[conn_list[i][0]]['centroid_pixel'][1]
            cy2 = hand_rp[conn_list[i][1]]['centroid_pixel'][1]
            
            
            # create color map where palm connects up into the hand
            if conn_list[i][0][0] == 'D' and conn_list[i][1][0] == 'P':
                plt.plot([cx1,cx2],[cy1,cy2],10,color=hand_rp[conn_list[i][1]]['color'])
            else:
                plt.plot([cx1,cx2],[cy1,cy2],10,color=hand_rp[conn_list[i][0]]['color'])


    # plot lines 
    plt.axis('equal') 
    plt.axis('off') 
    
    
# %%[create graph for hand mesh]

def mesh_as_graph(hand_rp):
    '''
    Makes a networkx graph network from the hand mesh.

    Parameters
    ----------
    hand_rp : Dict
        Hand region information.
        
    Returns
    -------
    hand_mesh : networkx graph
        graph with nodes for each hand centroid, and connections along the surface

    '''

    # get all region keys
    hand_tags = sorted(list(hand_rp.keys()))
    
    # create empty graph
    hand_mesh = nx.Graph()
    
    # add node for each region in key_list with centroid
    for i in range(len(hand_tags)):
        hand_mesh.add_node(hand_tags[i],pos=hand_rp[hand_tags[i]]['centroid_pixel'])
    
    # get distances between each node
    hand_distances = calc_hand_distances(hand_rp)
    
    connections = get_connections_list()
    
    # add all the connection distances between centroid pairs using lookup code
    # go through each list and plot
    for j in range(len(connections)):
        
        conn_list = connections[j]
        
        # loop through all combinations and plot for digit  
        for i in range(len(conn_list)):
            
            # get centroids for each in digit pair
            distance = hand_distances_pair(hand_distances,conn_list[i][0],conn_list[i][1])
            
            # add edges for this node
            hand_mesh.add_edge(conn_list[i][0],conn_list[i][1], weight=distance)

    return hand_mesh



# %%[connections array]

def get_connections_list():
    '''
    lists of connections between each of the hand centroids

    Returns
    -------
    digit_c : list
        connections between all the digits
        
    palm_c : list
        connections between all the palm regions
        
    digit_palm_c : list
        connections between digits and the palm

    '''
    
    digit_c = [['D1p','D1d'],['D2p','D2m'],['D2m','D2d'],['D3p','D3m'],['D3m','D3d'],
                ['D4p','D4m'],['D4m','D4d'],['D5p','D5m'],['D5m','D5d']]
    
    palm_c = [['Pw1','Pp2'],['Pw1','Pp1'],['Pw1','Pw2'],['Pp1','Pp2'],['Pw3','Pp1'],
              ['Pw2','Pw3'],['Pw3','Pw4'],['Pw4','Pp1'],['Pw2','Pp1'],['Pw4','Pp2']]
    
    digit_palm_c = [['D1p','Pw1'],['D2p','Pw2'],['D3p','Pw2'],['D3p','Pw3'],
                    ['D4p','Pw3'],['D4p','Pw4'],['D5p','Pw4']]
    
    return digit_c, palm_c, digit_palm_c
 
 
 
# %%[Draw graph]
def draw_graph(hand_graph):
    '''
    Plots the networkx graph network

    Parameters
    ----------
    hand_graph : networkx graph
        networkx graph of the hand connections

    Returns
    -------
    pos : dict
        Coordinate positions of each of the nodes
        
    edges : tuple
        Edge data with connections between each node
        
    weights : tuple
        strength of the connections (here the euclidean distances)

    '''
    edges,weights = zip(*nx.get_edge_attributes(hand_graph,'weight').items())
    
    pos = nx.get_node_attributes(hand_graph, 'pos')
    
    # nodes
    nx.draw_networkx_nodes(hand_graph, pos, node_color='b', node_size=300)
    
    # edges
    nx.draw_networkx_edges(hand_graph, pos, edge_color=weights, edge_cmap=plt.cm.Blues, edgelist=edges,
                            width=2)
    
    # labels
    nx.draw_networkx_labels(hand_graph, pos, font_size=7, font_color='w', font_family='sans-serif')
    
    plt.axis('equal')
    plt.axis('off')
    plt.show()
    
    return pos, edges, weights


# %%[Calculate shortest path on the graph]
def shortest_path_route(hand_graph):
    '''
    Calculates the shortest paths between all nodes in the network.
    Uses dijkstra distances from networkx

    Parameters
    ----------
    hand_graph : networkx graph
        networkx graph of the hand connections

    Returns
    -------
    dist_table : ndarray
        Distances between all the graph nodes (hand centroids)
        
    location_keys : str array
        tags of all the corresponding locations to the distance table
        
    dist_upper : ndarray
        Upper triangle of the distance array, rest are zero-ed.
        
    lengths : dict
        Dictionary with distances between every region

    '''
    # Calculate all path lengths
    lengths = dict(nx.all_pairs_dijkstra_path_length(hand_graph))
    
    # Convert to an array
    location_keys = np.sort(list(lengths.keys()))
    
    # create table
    dist_table = np.zeros((len(location_keys),len(location_keys)))
    
    # add data to table
    for i in range(len(location_keys)):
        for j in range(len(location_keys)):
            dist_table[i,j] = round(lengths[location_keys[i]][location_keys[j]]*10)
    
    # convert to integers
    dist_table = dist_table.astype(int)
    
    # get top part only
    dist_upper = np.triu(dist_table)
    
    # return distance table and keys
    return dist_table, location_keys, dist_upper, lengths


# %%[get mesh distance for a single pair of points in the graph network]
def mesh_distance_pair(region_1, region_2, hand_graph):
    '''
    distance between any two regions along the mesh

    Parameters
    ----------
    region_1 : str
        tag of the first region name eg. 'D1d'. Must be sub region tag.
        
    region_2 : str
        tag of the second region name eg. 'D2d'. Must be sub region tag.
        
    hand_graph : networkx graph
        graph of the hand mesh

    Returns
    -------
    distance : float
        distance between the two regions along the hand mesh

    '''
    assert len(region_1) == 3, "region tags must be sub region tags eg. 'D1d'"
    assert len(region_2) == 3, "region tags must be sub region tags eg. 'D1d'"
    
    # calculate all shortest paths along the mesh
    short_path = shortest_path_route(hand_graph)[3]
    
    # return shortest path distance for two regions
    distance = short_path[region_1][region_2]

    return distance

# %%[get mesh distance for all points in the graph network]   
 
def mesh_distance_all(hand_graph):
    '''
    Returns heatmap with the mesh distances

    Parameters
    ----------
    hand_graph : networkx graph
        graph of the hand mesh

    Returns
    -------
    None.

    '''
    
    # calculate all shortest paths along the mesh
    short_path = shortest_path_route(hand_graph)
    
    # get labels
    labels = short_path[1]
    
    # plot as heatmap
    sns.heatmap(short_path[0], yticklabels=labels,xticklabels=labels)
    plt.show()
    
