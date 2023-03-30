import torch
import numpy as np
from torch import nn
import math



class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim,
            graph_size,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim
        self.graph_size = graph_size

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))
        self.W_out = nn.Parameter(torch.Tensor(n_heads, val_dim, embed_dim))
        
        self.W_edge1 = nn.Linear(embed_dim, n_heads)
        self.tanh = nn.Tanh()
        self.init_parameters()
        

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, inputs, h=None, mask=None):
        """

        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        q, embeds = inputs
        e = embeds#.view(-1, self.embed_dim, self.graph_size, self.graph_size)
        
        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)
        #print(f"\nhflat : {hflat.shape}")
        #print(f"embeds : {e.shape}")

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)
        #print(f"shp_q : {shp_q}")

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)
        #print(f"(Q, K, V) : {Q.shape}, {K.shape}, {V.shape}")
        
        
        #Edge encoding
        e1 = self.W_edge1(e)
        e_comp = self.tanh(torch.movedim(e1, -1, 0))
        #print(f"e1 : {e_comp.shape}")
        

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3)) + e_comp
        #print(f"compatibility : {compatibility.shape}")

        # Optionally apply mask to prevent attention
        if mask is not None:
            #mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            mask = mask.view(1, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -np.inf

        attn = torch.softmax(compatibility, dim=-1)
        #print(f"attn : {attn.shape}")

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc

        heads = torch.matmul(attn, V)
        
        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        return [out, embeds]
    

    
    
################################################## COMPLEX MHA LAYERS #####################################################
    
class SkipConnection(nn.Module):
    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module
        
        if isinstance(module, MultiHeadAttention):
            self.GAT = True
        else:
            self.GAT = False

    def forward(self, inputs):
        
        if self.GAT:
            op = inputs[0] + self.module(inputs)[0]
            #print(f"MHA op: {op.shape}")
        else:
            op = inputs[0] + self.module(inputs[0])
            #print(f"Other op: {op.shape}")
        return [op, inputs[-1]]
    
    


class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalizer = normalizer_class(embed_dim, affine=True)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        # self.init_parameters()

    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, inputs):
        input = inputs[0]
        if isinstance(self.normalizer, nn.BatchNorm1d):
            return [self.normalizer(input.view(-1, input.size(-1))).view(*input.size()), inputs[-1]]
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return [self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1), inputs[-1]]
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return [input, inputs[-1]]


        
        
class MultiHeadAttentionLayer(nn.Sequential):

    def __init__(
            self,
            n_heads,
            embed_dim,
            graph_size,
            feed_forward_hidden=512,
            normalization='batch',
    ):
        super(MultiHeadAttentionLayer, self).__init__(
            SkipConnection(
                MultiHeadAttention(
                    n_heads,
                    embed_dim,
                    embed_dim,
                    graph_size
                )
            ),
            Normalization(embed_dim, normalization),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
            ),
            Normalization(embed_dim, normalization)
        )
        
        
########################################################## GRAPH ENCODER ###################################################


class GraphAttentionEncoder(nn.Module):
    def __init__(
            self,
            n_heads,
            embed_dim,
            n_layers,
            graph_size,
            node_dim=None,
            normalization='batch',
            feed_forward_hidden=512
    ):
        super(GraphAttentionEncoder, self).__init__()
        
        self.graph_size = graph_size
        # To map input to embedding space
        self.init_embed = nn.Linear(node_dim, embed_dim) if node_dim is not None else None

        self.layers = nn.Sequential(*(
            MultiHeadAttentionLayer(n_heads, embed_dim, graph_size, feed_forward_hidden, normalization)
            for _ in range(n_layers)
        ))

    def forward(self, inputs, mask=None):
        x = inputs[0]
        
        assert mask is None, "TODO mask not yet supported!"

        # Batch multiply to get initial embeddings of nodes
        h = self.init_embed(x.view(-1, x.size(-1))).view(*x.size()[:2], -1) if self.init_embed is not None else x

        h = self.layers(inputs)

        return (
            h[0],  # (batch_size, graph_size, embed_dim)
            h[0].mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )
