
# VRP gym Environment Class
# instance = {'node_features':__, 'edge_features':__, 'coordinates':__}
# For now assuming only the demands as edge features, without dynamism as in Kool et al implementation

import gym
from gym import spaces
import torch
import numpy as np
import collections

import sys
sys.path.append("../../realistic_vrp")
from matplotlib import pyplot as plt

from data.plot_vrp import *


class VRPEnv(gym.Env):

    def __init__(self, instance, idx=0):
        super(VRPEnv, self).__init__()
        
        self._coordinates = instance['coordinates']
        self._node_features = instance['node_features']
        self._edge_features = instance['edge_features']
        #self._dynamics = instance['dynamics']
        self.n_nodes = len(self._node_features)  #Not counting depot node
        self.env_id = idx
        self.state = None
        self.action_space = spaces.Discrete(self.n_nodes)
        self.observation_space = spaces.Dict({
            'node_features': spaces.Box(low=0, high=1.0, shape=(self._node_features.shape)),
            'edge_features': spaces.Box(low=0, high=1.0, shape=(self._edge_features.shape)),            
            'action_mask' :  spaces.MultiDiscrete([self.n_nodes for _ in range(self.n_nodes)]),
            'curr_pos_idx' : spaces.Discrete(self.n_nodes+2),
            'remaining_capacity' : spaces.Box(low=0, high=1.0),
            'ids' : spaces.Discrete(1)
        })



        
    def reset(self):
        self._action_mask = np.ones(self.n_nodes)
        self._curr_pos_idx = 0
        self._all_done = False
        self._vehicle_capacity = 1.0
        obs = {
            'node_features': self._node_features,
            'edge_features': self._edge_features,
            #'dynamics' : self._dynamics,
            
            'action_mask' : self._action_mask,
            'curr_pos_idx' : self._curr_pos_idx,
            'remaining_capacity' : self._vehicle_capacity,
            'ids' : self.env_id,
            }
        self.state = obs
        #print(f"Env: {self.env_id} reset")
        return obs

    
    
    def update_action_mask(self, action_mask, action):
        new_action_mask = np.copy(action_mask)
        if action > 0: #Only mask the node visit if its a customer, not for depot node
            new_action_mask[int(action)] = 0
        return new_action_mask



    
    def step(self, action):
        """ state: current state dict, action: chosen_idx; return: updated state_dict"""        
        chosen_node_idx = int(action)
                
        try:
            state = self.state.obs
        except:
            state = self.state
            
        remaining_capacity = state["remaining_capacity"]
        curr_pos_idx = state["curr_pos_idx"]
        action_mask = np.copy(state["action_mask"])
        reward = 0
        
        
        if action_mask[chosen_node_idx] != 0:
            if chosen_node_idx != 0:
                #print(state["node_features"][chosen_node_idx])
                used_capacity = round(float(state["node_features"][chosen_node_idx][2]), 2)
                reward = state["edge_features"][curr_pos_idx, chosen_node_idx] #Edge features has distance/time values directly at the specified indeces
                remaining_capacity = round(remaining_capacity - used_capacity, 2)
            else:
                remaining_capacity = self._vehicle_capacity
            
                #print(state["edge_features"][curr_pos_idx, chosen_node_idx])
                #print(f"from:{curr_pos_idx}, to:{chosen_node_idx}, reward:{reward}")
            
               
        action_mask = self.update_action_mask(action_mask, action)
        
        remaining = list(collections.Counter(action_mask).items())
        #print(remaining)
        if remaining[0][1] == 1:
            self._all_done = True
            #print(f"Env: {self.env_id}, all nodes visited!!")
        
             
        obs = {
            'node_features': self._node_features,
            'edge_features': self._edge_features,
            #'dynamics' : self._dynamics,
            
            'action_mask' : action_mask,
            'curr_pos_idx' : chosen_node_idx,
            'remaining_capacity' : remaining_capacity,
            'ids' : self.env_id,
            }
        
        self.state = obs
        bl_rew, bl_act = self.get_greedy_action_rew(state)
        info = {"bl_rew": bl_rew, "bl_act": bl_act}
        #print(f"Env: {self.env_id}, step for action: {curr_pos_idx}-{chosen_node_idx}, reward: {reward}, done: {self._all_done}, capacity: {remaining_capacity}")
        #print(obs, reward, self._all_done)
        return obs, reward, self._all_done, info
    
    
    
    def get_greedy_action_rew(self, batch_obs):
        #print(batch_obs)
        greedy_rews = []
        
        curr_pos = batch_obs["curr_pos_idx"]
        demands = batch_obs["node_features"][:,2]
        distances = batch_obs["edge_features"][curr_pos]
        mask_tensor = ~torch.tensor(batch_obs["action_mask"], dtype=torch.bool)
        action_order = np.array(torch.topk(distances, 3).indices)
            
        #print(distances)
        #print(action_order)
            
        n = int(action_order.shape[0])
        action = int(action_order[1])
            
        if action == 0: #If selected node is depot node, confirm if any step is possible to next probable node
            alt_action = int(action_order[2])
            if demands[alt_action] <= batch_obs["remaining_capacity"]:
                action = alt_action
                    
            if batch_obs["remaining_capacity"] < demands[action]:
                action = 0
            
        rew = torch.tensor(distances[action])
        #print(curr_pos, action)
        #print(rew)

        return rew, action
    
    
    def render(self, obs, tours):
        """
        Plots the customer nodes with the optimal routes selected
        """
        
        for i, (data, tour) in enumerate(zip(obs, tours)):
            fig, ax = plt.subplots(figsize=(10, 10))
            plot_vehicle_routes(data, tour, ax, visualize_demands=False, demand_scale=50, round_demand=True)