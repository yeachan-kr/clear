
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function

from attentions import MultiHeadSelfAttention, FFN
from activations import get_activation
from transformer_utils import *
# from torchmetrics.functional import pairwise_cosine_similarity

from typing import Dict, List, Optional, Set, Tuple, Union

class Adapter(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()

    def forward(self, x):
        return x

class BinaryLayer(Function):
    def forward(self, input):
        return torch.heaviside(input, torch.tensor([0.0]).to(input.device))

    def backward(self, grad_output):
        return grad_output.clamp_(-1, 1)

class BaseAdapter(Adapter):
    """ BaseAdapter Module (feed-forwad version) """
    def __init__(self, in_dim, hidden_dim):
        super().__init__(in_dim, hidden_dim)
        self.down_layer = nn.Linear(in_dim, hidden_dim)
        self.up_layer = nn.Linear(hidden_dim, in_dim)
        self.gelu = nn.GELU()

        self.hidden_states = None
        
    def forward(self, x):
        down_x = self.down_layer(x)
        hidden = self.gelu(down_x)
        up_x = self.up_layer(hidden) + x
        
        return up_x



class DebiasedAdapter(Adapter):
    """ BaseAdapter Module (feed-forwad version) """
    def __init__(self, in_dim, hidden_dim):
        super().__init__(in_dim, hidden_dim)
        self.down_layer = nn.Linear(in_dim, hidden_dim)
        self.up_layer = nn.Linear(hidden_dim, in_dim)
        self.gelu = nn.GELU()

        self.hidden_states = None
        
    def forward(self, x):
        down_x = self.down_layer(x)
        hidden = self.gelu(down_x)
        up_x = self.up_layer(hidden) + x
        
        return up_x


class AdaMixAdapter(Adapter):
    """ BaseAdapter Module (feed-forwad version) """
    def __init__(self, in_dim, hidden_dim, num_adapters=4):
        super().__init__(in_dim, hidden_dim)
        self.adamix_down = nn.Linear(in_dim, hidden_dim)

        self.adamix_up = nn.Parameter(torch.randn(num_adapters, hidden_dim, in_dim))
        nn.init.normal_(self.adamix_up, std=1e-2)
        
        self.adamix_up_bias = nn.Parameter(torch.randn(num_adapters, in_dim))
        nn.init.normal_(self.adamix_up_bias, std=1e-2)
        self.gelu = nn.GELU()

        self.sequence_code = None
        self.sequence_all = False

    def forward(self, x, routing=None):
        # random selection among two
        # [B, S, 2] -> [B, S, D, R]
        if self.sequence_all:
            down_x = self.adamix_down(x)
            hidden = self.gelu(down_x)

            up_layer = self.adamix_up.mean(dim=0)
            up_bias = self.adamix_up_bias.mean(dim=0)
            
            up_x = (hidden @ up_layer) + up_bias
        else:
            down_x = self.adamix_down(x)
            hidden = self.gelu(down_x)
            
            up_layer = self.adamix_up[self.sequence_code]
            up_bias = self.adamix_up_bias[self.sequence_code]

            up_x = hidden.unsqueeze(2) @ up_layer
            up_x = up_x.squeeze(2) + up_bias

        return up_x


class BaseAdapterTransformerBlock(nn.Module):
    def __init__(self, config, rank=16, nadapter=3):
        super().__init__()

        assert config.dim % config.n_heads == 0

        self.attention = MultiHeadSelfAttention(config)
        self.sa_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)

        self.ffn = FFN(config)
        self.output_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)

        self.adapter_set1 = [BaseAdapter(in_dim=config.dim, hidden_dim=rank) for _ in range(nadapter)]
        self.adapter_set1 = nn.ModuleList(self.adapter_set1)
        
        self.adapter_set2 = [BaseAdapter(in_dim=config.dim, hidden_dim=rank) for _ in range(nadapter)]
        self.adapter_set2 = nn.ModuleList(self.adapter_set2)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        adapter_idx = 0
    ) -> Tuple[torch.Tensor, ...]:
        """
        Parameters:
            x: torch.tensor(bs, seq_length, dim)
            attn_mask: torch.tensor(bs, seq_length)

        Returns:
            sa_weights: torch.tensor(bs, n_heads, seq_length, seq_length) The attention weights ffn_output:
            torch.tensor(bs, seq_length, dim) The output of the transformer block contextualization.
        """
        # Self-Attention
        sa_output = self.attention(
            query=x,
            key=x,
            value=x,
            mask=attn_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        if output_attentions:
            sa_output, sa_weights = sa_output  # (bs, seq_length, dim), (bs, n_heads, seq_length, seq_length)
        else:  # To handle these `output_attentions` or `output_hidden_states` cases returning tuples
            assert type(sa_output) == tuple
            sa_output = sa_output[0]

        adapter1 = self.adapter_set1[adapter_idx]
        sa_output = adapter1(sa_output)
        sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)

        # Feed Forward Network
        ffn_output = self.ffn(sa_output)  # (bs, seq_length, dim)

        adapter2 = self.adapter_set2[adapter_idx]
        ffn_output = adapter2(ffn_output)
        ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)

        output = (ffn_output,)
        if output_attentions:
            output = (sa_weights,) + output
        return output


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.dim % config.n_heads == 0

        self.attention = MultiHeadSelfAttention(config)
        self.sa_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)

        self.ffn = FFN(config)
        self.output_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Parameters:
            x: torch.tensor(bs, seq_length, dim)
            attn_mask: torch.tensor(bs, seq_length)
        Returns:
            sa_weights: torch.tensor(bs, n_heads, seq_length, seq_length) The attention weights ffn_output:
            torch.tensor(bs, seq_length, dim) The output of the transformer block contextualization.
        """
        # Self-Attention
        sa_output = self.attention(
            query=x,
            key=x,
            value=x,
            mask=attn_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        if output_attentions:
            sa_output, sa_weights = sa_output  # (bs, seq_length, dim), (bs, n_heads, seq_length, seq_length)
        else:  # To handle these `output_attentions` or `output_hidden_states` cases returning tuples
            assert type(sa_output) == tuple
            sa_output = sa_output[0]
        sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)

        # Feed Forward Network
        ffn_output = self.ffn(sa_output)  # (bs, seq_length, dim)
        ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)

        output = (ffn_output,)
        if output_attentions:
            output = (sa_weights,) + output
        return output


# class MultiLinear(nn.Module):
#     def __init__(self, nadapter=2, in_dim=768, out_dim=768):
#         super().__init__()
#         self.linears = [nn.Linear(in_dim, out_dim) for _ in range(nadapter)]
#         self.linears = nn.ModuleList(self.linears)
