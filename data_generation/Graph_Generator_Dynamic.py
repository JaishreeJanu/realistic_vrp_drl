# Importing required libraries
import subprocess

subprocess.call(['pip', 'install', 'bs4'])
import torch
from sklearn.preprocessing import minmax_scale

import overpy
import requests 
from bs4 import BeautifulSoup
from urllib import parse
import random
import pickle
import numpy as np
import pandas as pd


#setup OSRM 
from routingpy import OSRM
from routingpy.routers import options
options.default_timeout=None



class OSM_dynamic_graph():
    """
    For a given address query [city_name, shop_type], first we get co-ordinates and distance values from Open Street Map
    Then generate a dataset of n_instances having graph structure information (node and edge features)
    Other shop types : supermarket, convenience, clothes, hairdresser, car_repair, bakery
    """
    

    def __init__(self):
        self._coordinates = None
        self.nodes_df = None
        self.edge_matrix = None
        self.delay_matrix = None
    
    
    def get_area_code(self, address):
        """
        Gets the area code
        """
        url = "https://www.openstreetmap.org/geocoder/search_osm_nominatim?query=" + parse.quote(address)
        r = requests.get(url) 
        soup = BeautifulSoup(r.content, 'html5lib')
        osm_link = soup.find('a', attrs = {'class':'set_position'})
        relation_id = osm_link.get('data-id').strip()	
        return int(relation_id) + 3600000000 # 3600000000 is the offset in ids 
    
    
    
    def get_coordinates(self, address, shop_types):
        # Get all coordinates
        # More shop types https://wiki.openstreetmap.org/wiki/Key:shop
        
        area_code = self.get_area_code(address)
        api = overpy.Overpass()
        all_coordinates = []
        
        for shop_type in shop_types:

            request = api.query(f"""area({area_code});
            (node[shop={shop_type}](area);
            way[shop={shop_type}](area);
            rel[shop={shop_type}](area);
            ); out center;""")
    
            coords = [[float(node.lon), float(node.lat)] for node in request.nodes]
            coords += [[float(way.center_lon), float(way.center_lat)] for way in request.ways]
            coords += [[float(rel.center_lon), float(rel.center_lat)] for rel in request.relations]
            all_coordinates.extend(coords)
            
            print(f"Total {len(coords)} points found on the map for search query: {address, shop_type}")
        self._coordinates = all_coordinates
        return all_coordinates
    
    
            
    
    def generate_nodes(self, coordinates):
        
        #Normalization of co-ordinates -- values between 0 to 1
        norm_coordinates = minmax_scale(coordinates, feature_range=(0, 1), axis=0)
        
        # Non-uniform demand generation -- normalized random values between 0.10 to 0.50
        demands = minmax_scale(np.random.randint(low=10, high=50, size=(len(coordinates), 1)), feature_range=(0.1, 0.5), axis=0)
        
        # Creating a dataframe of nodes
        nodes_df = pd.DataFrame(list(coordinates))
        coordinates = nodes_df.apply(lambda row: row.values, axis=1)
        nodes_df = pd.DataFrame(coordinates, columns=['raw_coordinates']).reset_index()
        
        nodes_df["coordinates"] = norm_coordinates.tolist()
        nodes_df["demands"] = demands
        self.nodes_df = nodes_df
        return nodes_df
    
    
    
    def generate_edges(self, nodes_df, edge_type = "distance"):
        n_nodes = nodes_df["demands"].shape[0]
        n_max = 100
        n_blocks = int(n_nodes/n_max)
        
        sizes = [n_max for i in range(n_blocks)]
        edge_matrix = pd.DataFrame(index=range(n_blocks*n_max), columns=range(n_blocks*n_max))
        client = OSRM(base_url="https://router.project-osrm.org")
        
        for i, x_size in enumerate(sizes):
            x_start = i*n_max
            x_end = x_start + x_size
            
            for j, y_size in enumerate(sizes):
                y_start = j*n_max
                y_end = y_start + y_size
                #print(i, j, x_size, y_size, f"\ts{x_start}_{x_end} to e{y_start}_{y_end}")
                
                from_x_nodes = list(nodes_df["raw_coordinates"].iloc[x_start:x_end])
                to_y_nodes = list(nodes_df["raw_coordinates"].iloc[y_start:y_end])
                locations = from_x_nodes + to_y_nodes
                
                # Defining source and r=destination indeces
                sources = [idx for idx in range(len(from_x_nodes))]
                destinations = [idx for idx in range(len(from_x_nodes), len(locations))]
                
                # Getting distance matrix for selected block and storing in the edge matrix
                dist_matrix = client.matrix(locations=locations, sources=sources, destinations=destinations, profile="car")
                
                if edge_type == "distance":
                    edge_matrix.iloc[x_start:x_end, y_start:y_end] = np.array(dist_matrix.distances)
                elif edge_type == "time":
                    edge_matrix.iloc[x_start:x_end, y_start:y_end] = np.array(dist_matrix.durations)
        self.edge_matrix = edge_matrix
        return edge_matrix
    
        
    
    
    def generate_graphs(self, nodes_df, edge_matrix, delays_matrix, graph_size, n_instances):
        """
        Generates n_instances of given graph size, takes processed nodes_df and edge matrix as input
        """
        instances = []
        delay_times = [t for t in range(24)]
        

        for num in range (n_instances):
            # Node coordinates for the graph are obtained by randomly sampling graph_size nodes from all queried locations on OSM
            instance_data = nodes_df.loc[nodes_df["index"] < edge_matrix.shape[0]].sample(n = graph_size+1, random_state=np.random.seed())
            instance_indeces = list(instance_data["index"])
            dist_matrix = edge_matrix.iloc[instance_indeces, instance_indeces].to_numpy(dtype="float32")
            #print(f"--------------\n\ndist_matrix: {dist_matrix.shape}")
            
            
            # Generating stochastic delay masks for distance matrix dynamism
            for delay_time in delay_times:
                delay = delays_matrix.loc[delays_matrix["Time_Tag"] == delay_time].sample(1)["Delay"]
                delay = int(str(delay).split("%")[0].split()[-1]) / 100
                delay_mask = np.random.normal(loc=delay, scale=0.2*delay, size=(graph_size+1, graph_size+1))
                dist_delays = np.multiply(np.array(dist_matrix, dtype="float32"), delay_mask, dtype="float32")
                #print(delay)
                #print(f"delay_mask: {dist_delays.shape}")
                
                # Generating delayed edge features as dynamic rewards
                edge_features = torch.from_numpy(np.add(dist_matrix, dist_delays, dtype="float32"))
                edge_features = torch.div(edge_features, 1000) #Rewards are the distances in Km, i.e. transit cost to be minimized
                instance = {"coordinates": list(instance_data["raw_coordinates"]), "edge_features": edge_features, "norm_coordinates": list(instance_data["coordinates"]), "demands": list(instance_data["demands"]), "delay_time": delay_time, "delay": delay, "delay_mask": delay_mask}
                instances.append(instance)
                #print(f"edge_features: {edge_features}")
            
            if (num%1000) == 0:
                print(f"Generated {num} instances")
                
                
        

        # Now generating node features for all instances -- assigning first node as depot with no demand, 
        # Node features: [is_depot, is_customer, demand, coordinate_1, coordinate_2]
        
        all_graphs = []
        for num, instance in enumerate(instances):
            graph_nodes = []
            coordinates = instance["norm_coordinates"]
            
            
            for idx, node_coordinates in enumerate(coordinates):
                
                # Demand of current node
                demand = round(instance["demands"][idx], 2)
                
                if idx == 0:
                    node_features = np.append(np.array([1, 0, 0]), node_coordinates)
                    graph_nodes = node_features
                else:
                    node_features = np.append(np.array([0, 1, demand]), node_coordinates)
                    graph_nodes = np.vstack((graph_nodes, node_features))
                
            #print(graph_nodes)
            graph_nodes = torch.from_numpy(graph_nodes).float()
            graph_struct = {"node_features": graph_nodes, 
                            "edge_features": instance["edge_features"], 
                            "coordinates": instance["coordinates"], 
                            "delay": instance["delay"],
                            "delay_time": instance["delay_time"],
                            "delay_mask": instance["delay_mask"]}
            all_graphs.append(graph_struct)
            
                
        print(f"Generated {len(all_graphs)} graphs having {graph_size} customer nodes!")
        return all_graphs
    
    
    
    def save_data(self, file, file_path):
        with open(file_path, 'wb') as handle:
            pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            
    def load_data(self, file_path):
        with open(file_path, 'rb') as handle:
            data = pickle.load(handle)
            
        return data
    
    
    