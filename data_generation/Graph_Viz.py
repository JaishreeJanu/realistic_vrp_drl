#!/usr/bin/env python
# coding: utf-8


import os
import numpy as np
import pandas as pd
import torch


#get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt

from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# Code inspired by Google OR Tools plot:
# https://github.com/google/or-tools/blob/fb12c5ded7423d524fc6c95656a9bdc290a81d4d/examples/python/cvrptw_plot.py



# Extracting information about the test results by decoding the replay buffer
def decode_buffer(buffer):
    cols = ["env_id", "reward", "bl_reward", "curr_pos"]
    
    buffer_dict = {
        "env_id": buffer.info.env_id,
        "curr_pos": buffer.obs.curr_pos_idx,
        "curr_capacity": buffer.obs.remaining_capacity,
        "possible_actions": np.array([str(mask) for mask in buffer.obs.action_mask]),
        "action": buffer.act[:,0],
        "reward": buffer.rew,
        "bl_action": buffer.info.bl_act,
        "bl_reward": buffer.info.bl_rew,
        "done": buffer.done
    }
    
    # Create a dataframe with buffer observations
    buffer_df = pd.DataFrame.from_dict(buffer_dict)
    
    # Clean the buffer after removing unnecessary rows
    buffer_df = buffer_df.loc[(buffer_df[cols] != 0).any(axis=1)].reset_index()
    return buffer_df



def generate_tours(buffer_df, result):
    cols = ["env_id", "final_tour", "reward", "bl_reward"]
    tours = []
    bl_tours = []
    tour_lens = result["lens"]
    
    for idx, lens in enumerate(tour_lens):
        data = buffer_df.loc[buffer_df["env_id"] == idx]
        
        start_idxs = list(data["index"].loc[data["done"] == True].iloc[:-1])
        start_idxs = [i+1 for i in start_idxs]
        start_idxs.append(0)
        tour_start = max(start_idxs)
        
        end_idxs = list(data["index"].loc[data["done"] == True])
        tour_end = max(end_idxs)
        
        tour_data = data.loc[(data["index"] >= tour_start) & (data["index"] <= tour_end)]
        this_tour = tour_data["action"].tolist()
        this_tour_bl = tour_data["bl_action"].tolist()
        
        tours.append(this_tour)
        bl_tours.append(this_tour_bl)
        
    return tours, bl_tours


# Functions for tours visualization

def discrete_cmap(N, base_cmap=None):
    """
    Create an N-bin discrete colormap from the specified input map
    """
    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    
    return base.from_list(cmap_name, color_list, N)



def get_routes(graph_route):
    routes = []
    
    for idx, act in enumerate(graph_route):
        if idx == 0:
            this_route = [0, act]
            
        if int(act) != 0 and idx > 0:
            this_route.append(act)
            
        elif int(act) == 0:
            this_route.append(0)
            routes.append(np.array(this_route))
            this_route = [0]
            
    this_route.append(0)
    routes.append(np.array(this_route))
    return routes




def plot_vehicle_routes(graph_data, graph_route, ax1, markersize=5, visualize_demands=False, demand_scale=1, round_demand=False):
    """
    Plot the vehicle routes on matplotlib axis ax1.
    """
    
    # route is one sequence, separating different routes with 0 (depot)
    routes = get_routes(graph_route)
    print(routes)
    
    idx = graph_data['node_features'][:, 0].argmax().item()
    depot = graph_data['node_features'][0][3:]
    locs = graph_data['node_features'][1:][:,3:]
    demands = graph_data['node_features'][1:][:,2]
    capacity = demand_scale # Capacity is always 1
    
    x_dep, y_dep = depot
    ax1.plot(x_dep, y_dep, 'sk', markersize=markersize*4)
    #ax1.set_xlim(0, 1)
    #ax1.set_ylim(0, 1)
    
    legend = ax1.legend(loc='upper center')
    
    cmap = discrete_cmap(len(routes) + 2, 'nipy_spectral')
    dem_rects = []
    used_rects = []
    cap_rects = []
    qvs = []
    total_dist = 0
    for veh_number, r in enumerate(routes):
        color = cmap(len(routes) - veh_number) # Invert to have in rainbow order
        
        route_demands = demands[r - 1]
        coords = locs[r - 1, :]
        #print(r)
        #print(locs[r - 1, :])
        #trans_coords = coords.transpose(0,1)
        xs = coords[:, 0]
        ys = coords[:, 1]

        total_route_demand = sum(route_demands)
        assert total_route_demand <= capacity
        if not visualize_demands:
            ax1.plot(xs, ys, 'o', mfc=color, markersize=markersize, markeredgewidth=0.0)
        
        dist = 0
        x_prev, y_prev = x_dep, y_dep
        cum_demand = 0
        for (x, y), d in zip(coords, route_demands):
            dist += np.sqrt((x - x_prev) ** 2 + (y - y_prev) ** 2)
            
            cap_rects.append(Rectangle((x, y), 0.01, 0.1))
            used_rects.append(Rectangle((x, y), 0.01, 0.1 * total_route_demand / capacity))
            dem_rects.append(Rectangle((x, y + 0.1 * cum_demand / capacity), 0.01, 0.1 * d / capacity))
            
            x_prev, y_prev = x, y
            cum_demand += d
            
        dist += np.sqrt((x_dep - x_prev) ** 2 + (y_dep - y_prev) ** 2)
        total_dist += dist
        qv = ax1.quiver(
            xs[:-1],
            ys[:-1],
            xs[1:] - xs[:-1],
            ys[1:] - ys[:-1],
            scale_units='xy',
            angles='xy',
            scale=1,
            color=color,
            label='R{}, # {}, c {} / {}, d {:.2f}'.format(
                veh_number, 
                len(r), 
                int(total_route_demand) if round_demand else total_route_demand, 
                int(capacity) if round_demand else capacity,
                dist
            )
        )
        
        qvs.append(qv)
        
    ax1.set_title('{} routes, total distance {:.2f}'.format(len(routes), total_dist))
    ax1.legend(handles=qvs)
    
    pc_cap = PatchCollection(cap_rects, facecolor='whitesmoke', alpha=1.0, edgecolor='lightgray')
    pc_used = PatchCollection(used_rects, facecolor='lightgray', alpha=1.0, edgecolor='lightgray')
    pc_dem = PatchCollection(dem_rects, facecolor='black', alpha=1.0, edgecolor='black')
    
    if visualize_demands:
        ax1.add_collection(pc_cap)
        ax1.add_collection(pc_used)
        ax1.add_collection(pc_dem)