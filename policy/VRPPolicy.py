import numpy as np
import torch
from tianshou.policy import BasePolicy
from numba import njit

from typing import Any, Callable, List, Optional, Tuple, Union, Dict
from tianshou.env import DummyVectorEnv
from tianshou.data import Batch, ReplayBuffer, to_torch, to_torch_as


class REINFORCEPolicy(BasePolicy):
    """Implementation of REINFORCE algorithm."""
    def __init__(self, model: torch.nn.Module, optim: torch.optim.Optimizer,):
        super().__init__()
        self.actor = model
        self.optim = optim
        # action distribution
        self.dist_fn = torch.distributions.Categorical
    
    def get_obs(self, batch: Batch) -> Batch:
        try:
            batch_obs = batch.obs
        except:
            batch_obs = batch
        return batch_obs
    
        
    def forward(self, batch: Batch) -> Batch:
        """Compute action over the given batch data."""
        batch_obs = self.get_obs(batch)
        logits = self.actor(batch_obs).cpu()
        #print(f"logits: {logits}")
        dist = self.dist_fn(logits=logits)
        #print(dist.probs)
        act = self.get_actions(dist, batch_obs)
        #print(batch["action_mask"])
        #print(batch["remaining_capacity"])
        #print(act)
        return Batch(act=act, dist=dist)


    def process_fn(self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray) -> Batch:
        """Compute the discounted returns for each transition."""
        #print(batch)
        returns, bl_returns = self.compute_episodic_return(batch, buffer, indices, gamma=0.99, gae_lambda=1.0)
        batch.returns = returns
        batch.bl_returns = bl_returns
        #print(f"pre-batch computed returns: {batch.bl_returns}")
        return batch
        


    def learn(self, batch: Batch, batch_size: int, repeat: int) -> Dict[str, List[float]]:
        """Perform the back-propagation."""
        logging_losses = []
        for _ in range(repeat):
            for minibatch in batch.split(batch_size, merge_last=True):
                self.optim.zero_grad()
                result = self(minibatch)
                dist = result.dist
                act = to_torch_as(minibatch.act, result.act)
                ret = to_torch(minibatch.returns, torch.float)
                bl_ret = to_torch(minibatch.bl_returns, torch.float)
                log_prob = dist.log_prob(act).reshape(len(ret), -1).transpose(0, 1)
                loss = (log_prob * (ret - bl_ret)).mean()
                loss.backward()
                self.optim.step()
                logging_losses.append(loss.item())
        return {"loss": logging_losses}
    
    
    def update(self, sample_size: int, buffer: Optional[ReplayBuffer],
               **kwargs: Any) -> Dict[str, Any]:
        """Update the policy network and replay buffer.
        """
        if buffer is None:
            return {}
        batch, indices = buffer.sample(sample_size)
        self.updating = True
        #print(batch)
        batch = self.process_fn(batch, buffer, indices)
        result = self.learn(batch, **kwargs)
        self.post_process_fn(batch, buffer, indices)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.updating = False
        return result
    
    
    
    def get_actions(self, dist, batch_obs):
        actions = []
        for idx, (probs, mask) in enumerate(zip(dist.probs, batch_obs["action_mask"])):
            demands = batch_obs["node_features"][idx,:,2]
            mask_tensor = ~torch.tensor(mask, dtype=torch.bool)
            action_order = np.array(torch.topk(probs.cpu(), 3).indices)
            
            #print(probs)
            #print(action_order)
            
            n = int(action_order.shape[1])
            action = int(action_order[0,0])
            
            if action == 0: #If selected node is depot node, confirm if any step is possible to next probable node
                alt_action = int(action_order[0,1])
                if demands[alt_action] <= batch_obs["remaining_capacity"][idx]:
                    action = alt_action
                    
            if batch_obs["remaining_capacity"][idx] < demands[action]:
                action = 0
                    
            #print(action)
            actions.append(action)
        actions = torch.tensor(actions).view(-1, 1)
        return actions
    
    
    
    def compute_episodic_return(self,
        batch: Batch,
        buffer: ReplayBuffer,
        indices: np.ndarray,
        v_s_: Optional[Union[np.ndarray, torch.Tensor]] = None,
        v_s: Optional[Union[np.ndarray, torch.Tensor]] = None,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute returns over given batch."""
        
        #print(f"to compute:{batch}")
        rew = batch.rew
        bl_rew = np.array(batch.info.bl_rew)
        
        end_flag = batch.done
        end_flag[np.isin(indices, buffer.unfinished_index())] = True
        #end_flag = np.logical_or(batch.terminated, batch.truncated)
        
        
        returns = _gae_return(rew, end_flag, gamma, gae_lambda)
        bl_returns = _gae_return(bl_rew, end_flag, gamma, gae_lambda)
        
        return returns, bl_returns
    
    
@njit
def _gae_return(
    rew: np.ndarray,
    end_flag: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> np.ndarray:
    returns = np.zeros(rew.shape)
    delta = rew * gamma
    discount = (1.0 - end_flag) * (gamma * gae_lambda)
    gae = 0.0
    for i in range(len(rew) - 1, -1, -1):
        gae = delta[i] + discount[i] * gae
        returns[i] = gae
    return returns  