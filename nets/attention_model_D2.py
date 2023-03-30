import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import math
from typing import NamedTuple
from utils.tensor_functions import compute_in_batches

from nets.graph_encoder_D2 import GraphAttentionEncoder
from torch.nn import DataParallel
from utils.beam_search import CachedLookup
from utils.functions import sample_many
from utils.boolmask import get_mask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def get_finished(batch_obs):
    return batch_obs['visited'].sum(-1) == batch_obs['visited'].size(-1)

def all_finished(batch_obs):            
    return batch_obs['i'].item() >= batch_obs['demand'].size(-1) and batch_obs['visited'].all()


def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)

    
    
    

################################################################## ATTENTION MODEL #####################################################

class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)
        return AttentionModelFixed(
            node_embeddings=self.node_embeddings[key],
            context_node_projected=self.context_node_projected[key],
            glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
            glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
            logit_key=self.logit_key[key]
        )

    

class AttentionModel(nn.Module):

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 graph_size,
                 n_encode_layers=2,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization='batch',
                 n_heads=8,
                 checkpoint_encoder=False,
                 shrink_size=None,
                 decode_type='sampling'):
        super(AttentionModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = decode_type
        self.temp = 1.0
        self.graph_size = graph_size+1


        self.tanh_clipping = tanh_clipping

        self.mask_inner = mask_inner
        self.mask_logits = mask_logits


        self.n_heads = n_heads
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size


        # Embedding of last node + remaining_capacity / remaining length / remaining prize to collect
        step_context_dim = embedding_dim + 1
        node_dim = 5  # x, y, demand / prize
        
        # Special embedding projection for depot node
        self.init_embed = nn.Linear(node_dim, embedding_dim) # to check
        self.init_embed_edge = nn.Linear(1, embedding_dim)

        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            graph_size = self.graph_size,
            n_layers=self.n_encode_layers,
            normalization=normalization
        )

        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_step_context = nn.Linear(step_context_dim, embedding_dim, bias=False)
        assert embedding_dim % n_heads == 0
        # Note n_heads * val_dim == embedding_dim so batch_obs to project_out is embedding_dim
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp

    def forward(self, batch_obs):
        """
        :param batch_obs: (batch_size, graph_size, node_dim) batch_obs node features or dictionary with multiple tensors
        :param return_pi: whether to return the output sequences, this is optional as it is not compatible with
        using DataParallel as the results may be of different lengths on different GPUs
        :return:
        """
        nodes_x = batch_obs["node_features"].to(device)
        batch_size, graph_size, node_dim = nodes_x.size()
        
        edges_x = torch.tensor(batch_obs["edge_features"], dtype=torch.float32).view(batch_size, graph_size, graph_size, -1).to(device)
        
        
        if self.checkpoint_encoder and self.training:  # Only checkpoint if we need gradients
            embeddings, _ = checkpoint(self.embedder, self.init_embed(nodes_x))
        else:
            node_embeddings = self.init_embed(nodes_x)
            edge_embeddings = self.init_embed_edge(edges_x)            
            
            
            embeddings, _ = self.embedder([node_embeddings, edge_embeddings])
        
        logits = self._inner(batch_obs, embeddings)

        return logits

    
    
    def beam_search(self, *args, **kwargs):
        return self.problem.beam_search(*args, **kwargs, model=self)

    def precompute_fixed(self, batch_obs):
        embeddings, _ = self.embedder(self._init_embed(batch_obs))
        # Use a CachedLookup such that if we repeatedly index this object with the same index we only need to do
        # the lookup once... this is the case if all elements in the batch have maximum batch size
        return CachedLookup(self._precompute(embeddings))

    def propose_expansions(self, beam, fixed, expand_size=None, normalize=False, max_calc_batch_size=4096):
        # First dim = batch_size * cur_beam_size
        log_p_topk, ind_topk = compute_in_batches(
            lambda b: self._get_log_p_topk(fixed[b.ids], b.batch_obs, k=expand_size, normalize=normalize),
            max_calc_batch_size, beam, n=beam.size()
        )

        assert log_p_topk.size(1) == 1, "Can only have single step"
        # This will broadcast, calculate log_p (score) of expansions
        score_expand = beam.score[:, None] + log_p_topk[:, 0, :]

        # We flatten the action as we need to filter and this cannot be done in 2d
        flat_action = ind_topk.view(-1)
        flat_score = score_expand.view(-1)
        flat_feas = flat_score > -1e10  # != -math.inf triggers

        # Parent is row idx of ind_topk, can be found by enumerating elements and dividing by number of columns
        flat_parent = torch.arange(flat_action.size(-1), out=flat_action.new()) // ind_topk.size(-1)

        # Filter infeasible
        feas_ind_2d = torch.nonzero(flat_feas)

        if len(feas_ind_2d) == 0:
            # Too bad, no feasible expansions at all :(
            return None, None, None

        feas_ind = feas_ind_2d[:, 0]

        return flat_parent[feas_ind], flat_action[feas_ind], flat_score[feas_ind]

    def _calc_log_likelihood(self, _log_p, a, mask):

        # Get log_p corresponding to selected actions
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)

        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        if mask is not None:
            log_p[mask] = 0

        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        return log_p.sum(1)

    

    def _inner(self, batch_obs, embeddings):

        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        fixed = self._precompute(embeddings)
        batch_size = batch_obs['ids'].size
        
        logits, _ = self._get_log_p(fixed, batch_obs)
        
        return logits

        ## Perform decoding steps
        #if not (self.shrink_size is None and all_finished(batch_obs)):
#
        #        unfinished = torch.nonzero(get_finished(batch_obs)== 0)
        #        unfinished = unfinished[:, 0]
        #        # Check if we can shrink by at least shrink_size and if this leaves at least 16
        #        # (otherwise batch norm will not work well and it is inefficient anyway)
        #        if 16 <= len(unfinished) <= batch_obs.ids.size(0) - self.shrink_size:
        #            # Filter batch_obss
        #            batch_obs = batch_obs[unfinished]
        #            fixed = fixed[unfinished]
#
        #    log_p, mask = self._get_log_p(fixed, batch_obs)
#
        #    # Select the indices of the next nodes in the sequences, result (batch_size) long
        #    selected = self._select_node(log_p.exp()[:, 0, :], mask[:, 0, :])  # Squeeze out steps dimension
        #
        #
        #    if self.shrink_size is not None and batch_obs['ids'].size < batch_size:
        #        log_p_, selected_ = log_p, selected
        #        log_p = log_p_.new_zeros(batch_size, *log_p_.size()[1:])
        #        selected = selected_.new_zeros(batch_size)
#
        #        log_p[batch_obs['ids'][:, 0]] = log_p_
        #        selected[batch_obs['ids'][:, 0]] = selected_        
#
#
#
        #return selected, mask
#
    
    
    def sample_many(self, batch_obs, batch_rep=1, iter_rep=1):
        """
        :param batch_obs: (batch_size, graph_size, node_dim) batch_obs node features
        :return:
        """
        # Bit ugly but we need to pass the embeddings as well.
        # Making a tuple will not work with the problem.get_cost function
        return sample_many(
            lambda batch_obs: self._inner(*batch_obs),  # Need to unpack tuple into arguments
            lambda batch_obs, pi: self.problem.get_costs(batch_obs[0], pi),  # Don't need embeddings as batch_obs to get_costs
            (batch_obs, self.embedder(self._init_embed(batch_obs))[0]),  # Pack batch_obs with embeddings (additional batch_obs)
            batch_rep, iter_rep
        )


    
    
    def _get_log_p_topk(self, fixed, batch_obs, k=None, normalize=False):
        log_p, _ = self._get_log_p(fixed, batch_obs, normalize=normalize)

        # Return topk
        if k is not None and k < log_p.size(-1):
            return log_p.topk(k, -1)

        # Return all, note different from torch.topk this does not give error if less than k elements along dim
        return (
            log_p,
            torch.arange(log_p.size(-1), device=log_p.device, dtype=torch.int64).repeat(log_p.size(0), 1)[:, None, :]
        )

    
    
    def _get_log_p(self, fixed, batch_obs, normalize=False):

        # Compute query = context node embedding
        step_context = self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings, batch_obs)).view(len(batch_obs), 1, -1)
        
        query = fixed.context_node_projected + step_context
        
        #print(f"\n\nfixed_context: {fixed.context_node_projected.shape}")
        #print(f"step_context: {step_context.shape}")
        #print(f"query: {query.shape}")
        
        # Compute keys and values for the nodes
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed, batch_obs)

        # Compute the mask
        # mask = get_mask(batch_obs)
        mask = batch_obs["action_mask"]

        # Compute logits (unnormalized log_p)
        logits, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask)
        
        if normalize:
            log_p = torch.log_softmax(logits / self.temp, dim=-1)

        assert not torch.isnan(logits).any()

        return logits, mask

        

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):

        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        # Compute the glimpse, rearrange dimensions so the dimensions are (n_heads, batch_size, num_steps, 1, key_size)
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_steps, graph_size)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        #print(f"\ncompatibility: {compatibility}")
        
        
        #print(f"\nmask info: {mask.shape}")
              
        if self.mask_inner:
            assert self.mask_logits, "Cannot mask inner without masking logits"
            mask_tensor = ~torch.tensor(mask, dtype=torch.bool)
            compatibility[mask_tensor[None, :, None, None, :].expand_as(compatibility)] = -math.inf
        
        #print(f"\ncompatibility-after_mask: {compatibility}")

        # Batch matrix multiplication to compute heads (n_heads, batch_size, num_steps, val_size)
        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)

        # Project to get glimpse/updated context node embedding (batch_size, num_steps, embedding_dim)
        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size))

        # Now projecting the glimpse is not needed since this can be absorbed into project_out
        # final_Q = self.project_glimpse(glimpse)
        final_Q = glimpse
        #print(f"final_Q: {final_Q.shape}")
        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        # logits = 'compatibility'
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))
        
        
        # From the logits compute the probabilities by clipping, masking and softmax
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
            #print(f"logits: {logits}")
            
        if self.mask_logits:
            #print(f"\nmask: {mask_tensor[:, None,  :]}")
            logits[mask_tensor[:, None,  :]] = -math.inf
            
            #print(f"logits-after_mask: {logits}")
        return logits, glimpse.squeeze(-2)

    
    
    
    def _get_attention_node_data(self, fixed, batch_obs):

        # TSP or VRP without split delivery
        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

    
    
    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )

    
    
    def _get_parallel_step_context(self, embeddings, batch_obs, from_depot=False):
        """
        Returns the context per step, optionally for multiple steps at once (for efficient evaluation of the model)
        
        :param embeddings: (batch_size, graph_size, embed_dim)
        :param prev_a: (batch_size, num_steps)
        :param first_a: Only used when num_steps = 1, action of first step or None if first step
        :return: (batch_size, num_steps, context_dim)
        """

        current_node = torch.Tensor(batch_obs['curr_pos_idx']).type(torch.int64).to(device)
        batch_size = batch_obs['ids'].size
        num_steps = 1
        
        idx = current_node.contiguous().view(batch_size, num_steps, 1).expand(batch_size, num_steps, embeddings.size(-1))
        
        node_context = torch.gather(
                        embeddings,
                         1,
                         idx
                     ).view(batch_size, embeddings.size(-1))
        
        ## Embedding of previous node + remaining capacity
        #if from_depot:
        #    # 1st dimension is node idx, but we do not squeeze it since we want to insert step dimension
        #    # i.e. we actually want embeddings[:, 0, :][:, None, :] which is equivalent
        #    return torch.cat(
        #        (
        #            embeddings[:, 0:1, :].expand(batch_size, num_steps, embeddings.size(-1)),
        #            # used capacity is 0 after visiting depot
        #            torch.Tensor(batch_obs['remaining_capacity'])
        #        ),
        #        -1
        #    )
        #else:
        
        return torch.cat((node_context,
                torch.Tensor(batch_obs['remaining_capacity']).view(-1, 1).to(device)),
                -1)
    
    
    def _select_node(self, probs, mask):

        assert (probs == probs).all(), "Probs should not contain any nans"

        if self.decode_type == "greedy":
            _, selected = probs.max(1)
            assert not mask.gather(1, selected.unsqueeze(
                -1)).data.any(), "Decode greedy: infeasible action has maximum probability"

        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)

            # Check if sampling went OK, can go wrong due to bug on GPU
            # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print('Sampled bad values, resampling!')
                selected = probs.multinomial(1).squeeze(1)

        else:
            assert False, "Unknown decode type"
        return selected

    
    
    def _precompute(self, embeddings, num_steps=1):

        # The fixed context projection of the graph embedding is calculated only once for efficiency
        graph_embed = embeddings.mean(1)
        # fixed context = (batch_size, 1, embed_dim) to make broadcastable with parallel timesteps
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]

        # The projection of the node embeddings for the attention is calculated once up front
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)

        # No need to rearrange key for logit as there is a single head
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous()
        )
        return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)