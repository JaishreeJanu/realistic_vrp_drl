
# VRP gym Environment Class
# instance = {'node_features':__, 'edge_features':__, 'coordinates':__}
# For now assuming only the demands as edge features, without dynamism as in Kool et al implementation

import gym
from gym import spaces
import torch
import numpy as np
import collections

import sys
import math
sys.path.append("../../realistic_vrp")
from matplotlib import pyplot as plt


class VRPEnv(gym.Env):

    def __init__(self, instance, idx=0):
        super(VRPEnv, self).__init__()
        
        self._coordinates = instance['coordinates']
        self._node_features = instance['node_features']
        self._edge_features = instance['edge_features']
        self.n_nodes = len(self._node_features)  #Not counting depot node
        self.env_id = idx
        
        self.state = None
        self.action_space = spaces.Discrete(self.n_nodes)
        self.observation_space = spaces.Dict({
            'node_features': spaces.Box(low=0, high=1.0, shape=(self._node_features.shape)),
            'edge_features': spaces.Box(low=0, high=1.0, shape=(self._edge_features.shape)), 
            
            # Vehicle state and active actions mask for model
            'action_mask' :  spaces.MultiDiscrete([self.n_nodes for _ in range(self.n_nodes)]),
            'curr_pos_idx' : spaces.Discrete(self.n_nodes+2),
            'remaining_capacity' : spaces.Box(low=0, high=1.0, shape=(1,)),
            
            
            # Vehicle state parameters for baseline model
            'bl_action_mask' :  spaces.MultiDiscrete([self.n_nodes for _ in range(self.n_nodes)]),
            'bl_curr_pos_idx' : spaces.Discrete(self.n_nodes+2),
            'bl_remaining_capacity' : spaces.Box(low=0, high=1.0, shape=(1,)),
            
            
            # Env id and completion status
            'ids' : spaces.Discrete(1),
            'all_done' : spaces.Discrete(2),
            'all_done_bl' : spaces.Discrete(2)
        })



        
    def reset(self):
        # Reset environment state
        self._vehicle_capacity = 1.0
        self.complete = False
        self._depot_pos = 0
        self._action_mask = np.ones(self.n_nodes)
        self.bl_action_mask = np.ones(self.n_nodes)
        
        obs = {
            'node_features': self._node_features,
            'edge_features': self._edge_features,  
            
            'action_mask' : self._action_mask,
            'curr_pos_idx' : self._depot_pos,
            'remaining_capacity' : self._vehicle_capacity,
            
            'bl_action_mask' : self.bl_action_mask,
            'bl_curr_pos_idx' : self._depot_pos,
            'bl_remaining_capacity' : self._vehicle_capacity,
            
            'ids' : self.env_id,
            'all_done' : 0,
            'all_done_bl' : 0
            }
        self.state = obs
        #print(f"Env: {self.env_id} reset")
        return obs

    



    
    def step(self, action):
        # Take an action and return updated state
        """ state: current state dict, action: chosen_idx; return: updated state_dict"""        
        action = int(action)
        
        try:
            state = self.state.obs
        except:
            state = self.state
        
        # Environment step and updated values after taking chosen action
        if state['all_done'] == 0:
            reward, action_mask, curr_pos_idx, remaining_capacity, _all_done = self._step(state, action)
        else:
            reward, action_mask, curr_pos_idx, remaining_capacity, _all_done = self.same_state(state)
        
        
        # Greedy Rollout Baseline model steps and state updates
        if state['all_done_bl'] == 0:
            bl_reward, bl_action, bl_action_mask, bl_curr_pos_idx, bl_remaining_capacity, _all_done_bl = self.bl_step(state)
        else:
            bl_action = 0
            bl_reward, bl_action_mask, bl_curr_pos_idx, bl_remaining_capacity, _all_done_bl = self.same_state(state, is_base = True)
        
        if _all_done == True and _all_done_bl == True:
            self.complete = True
        
        
        obs = {
            'node_features': self._node_features,
            'edge_features': self._edge_features,  
            
            'action_mask' : action_mask,
            'curr_pos_idx' : curr_pos_idx,
            'remaining_capacity' : remaining_capacity,
            
            'bl_action_mask' : bl_action_mask,
            'bl_curr_pos_idx' : bl_curr_pos_idx,
            'bl_remaining_capacity' : bl_remaining_capacity,
            
            'ids' : self.env_id,
            'all_done' : _all_done,
            'all_done_bl' : _all_done_bl
            }
        
        self.state = obs
        pos = state["curr_pos_idx"]
        bl_pos = state["bl_curr_pos_idx"]
        info = {"bl_rew": bl_reward, "bl_act": bl_action}
        
        # Debugging
        #if self.env_id == 0:
            #print(f"\nEnv: {self.env_id}, step for action: {pos}-{action}, reward: {reward}, done: {_all_done}, capacity: {remaining_capacity}, action_mask: {action_mask}")
            #print(f"Baseline Env: {self.env_id}, step for action: {bl_pos}-{bl_action}, reward: {bl_reward}, done: {_all_done_bl}, capacity: {bl_remaining_capacity}, 'bl_action_mask' : {bl_action_mask}")
        #print("Env", self.env_id, self.complete)
        
        return obs, reward, self.complete, info
    
    
    
    
    def same_state(self, state, is_base=False):
        rew = 0
        
        if is_base:
            action_mask = state['bl_action_mask']
            curr_pos_idx = state['bl_curr_pos_idx']
            remaining_capacity = state['bl_remaining_capacity']
            _all_done = state['all_done_bl']
        else:
            action_mask = state['action_mask']
            curr_pos_idx = state['curr_pos_idx']
            remaining_capacity = state['remaining_capacity']
            _all_done = state['all_done']
        
        return rew, action_mask, curr_pos_idx, remaining_capacity, _all_done
    
    
    
    
    def update_action_mask(self, action_mask, action):
        new_action_mask = np.copy(action_mask)
        if action > 0: #Only mask the node visit if its a customer, not for depot node
            new_action_mask[int(action)] = 0
        return new_action_mask
    
    
    
    
    def _step(self, state, action):
        """ state: current state dict, action: chosen_idx; return: updated state_dict"""        
        chosen_node_idx = int(action)
        remaining_capacity = state["remaining_capacity"]
        curr_pos_idx = state["curr_pos_idx"]
        action_mask = np.copy(state["action_mask"])
        reward = 0
        #print(f"env action: {action}, old action mask: {action_mask}")
        
        if action_mask[chosen_node_idx] != 0:
            if chosen_node_idx != 0:
                used_capacity = round(float(state["node_features"][chosen_node_idx][2]), 2)
                reward = state["edge_features"][curr_pos_idx, chosen_node_idx]
                remaining_capacity = round(remaining_capacity - used_capacity, 2)
            else:
                remaining_capacity = self._vehicle_capacity            
               
        action_mask = self.update_action_mask(action_mask, action)
        #print(f"new env action mask: {action_mask}\n\n")
        remaining = list(collections.Counter(action_mask).items())
        
        if remaining[0][1] == 1:
            all_done = 1
        else:
            all_done = 0
                
        return reward, action_mask, chosen_node_idx, remaining_capacity, all_done
    
    
    
    
    def bl_step(self, state):
        # Greedy Rollout Baseline model steps and state updates
        """ state: current state dict, action: chosen_idx; return: updated state_dict"""        
            
        remaining_capacity = state["bl_remaining_capacity"]
        curr_pos_idx = state["bl_curr_pos_idx"]
        action_mask = np.copy(state["bl_action_mask"])
        reward = 0
        
        bl_reward, bl_action = self.get_greedy_action_rew(state, action_mask, remaining_capacity, curr_pos_idx)
        #print(f"baseline action: {bl_action}, old action mask: {action_mask}")
        chosen_node_idx = bl_action

        if action_mask[chosen_node_idx] != 0:
            if chosen_node_idx != 0:
                used_capacity = round(float(state["node_features"][chosen_node_idx][2]), 2)
                remaining_capacity = round(remaining_capacity - used_capacity, 2)
            else:
                remaining_capacity = self._vehicle_capacity            
               
        action_mask = self.update_action_mask(action_mask, bl_action)
        #print(f"new baseline action mask: {action_mask}\n\n")
        remaining = list(collections.Counter(action_mask).items())
        
        if remaining[0][1] == 1:
            all_done = 1
        else:
            all_done = 0
        
        #nodes = state["node_features"][:,-2:]
        #print(f"env:{self.env_id}")
        #print(f"updated action mask:{action_mask}")
        #print(f"baseline rew, act: {bl_reward, bl_action}")
                
        return bl_reward, bl_action, action_mask, chosen_node_idx, remaining_capacity, all_done
    
    
    
    
    
    def get_greedy_action_rew(self, batch_obs, action_mask, remaining_capacity, curr_pos):
        #print(f"\nComputing greedy actions for curr_pos {curr_pos} and mask {action_mask}")
        #print(batch_obs)        
        
        demands = batch_obs["node_features"][:,2]
        distances = torch.clone(batch_obs["edge_features"][curr_pos])
        #print(f"distances: {distances}")
        mask_tensor = ~torch.tensor(action_mask, dtype=torch.bool)
        distances[mask_tensor] = 100000
        
        
        action_order = np.array(torch.topk(distances, 3, largest=False).indices)
        #print(f"distances with mask: {distances}")
        #print(f"action_order: {action_order}")
            
        n = int(action_order.shape[0])
        action = int(action_order[0])
            
        if action == 0: #If selected node is depot node, confirm if any step is possible to next probable node
            alt_action = int(action_order[1])
            if demands[alt_action] <= remaining_capacity:
                action = alt_action
                    
        if remaining_capacity < demands[action]:
            action = 0
            
        rew = distances[action]
        if rew == -math.inf or action == 0:
            rew = 0
            
        rew = float(rew)
        #print(curr_pos, action)
        #print(f"env {self.env_id}, greedy rew: {rew}")
        #print(f"selected greedy action: {action}")

        return rew, action

            
    def seed(self, seed):
        np.random.seed(seed)