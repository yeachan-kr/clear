
import math
import random
import numpy as np

from turtle import forward

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import InputFeatures

from attentions import MultiHeadSelfAttention, FFN
from activations import get_activation
from transformer_utils import *

from typing import Dict, List, Optional, Set, Tuple, Union
from blocks import TransformerBlock, BaseAdapterTransformerBlock, AdaMixAdapter
from transformers.modeling_outputs import BaseModelOutput

from transformers.models.bert.modeling_bert import BertEncoder, BaseModelOutputWithPoolingAndCrossAttentions, \
    apply_chunking_to_forward, BertAttention, BertIntermediate, BertOutput, Optional, Tuple, Union, BaseModelOutputWithPastAndCrossAttentions, logger, BertLayer


class Adapter(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()

    def forward(self, x):
        return x

class BaseAdapter(Adapter):
    """ BaseAdapter Module (feed-forwad version) """
    def __init__(self, in_dim, hidden_dim):
        super().__init__(in_dim, hidden_dim)
        self.down_layer = nn.Linear(in_dim, hidden_dim)
        self.up_layer = nn.Linear(hidden_dim, in_dim)
        self.gelu = nn.GELU()

        self.hidden_states = None
        self.routing_probs = None
        
    def forward(self, x):
        down_x = self.down_layer(x)
        hidden = self.gelu(down_x)
        up_x = self.up_layer(hidden) + x
        return up_x


class StochasticAdapter(Adapter):
    def __init__(self, in_dim, hidden_dim):
        super().__init__(in_dim, hidden_dim)
        self.down_layer = nn.Linear(in_dim, hidden_dim)
        self.up_layer = nn.Linear(hidden_dim, in_dim)
        self.gelu = nn.GELU()

        self.routing_probs = nn.Parameter(torch.rand(32)) # TODO batchsize
        self.routing_probs.requires_grad = False
        self.start_routing = False
        self.inference = False
        
    def forward(self, x):
        mask = self.routing_probs.view(-1, 1, 1)
        
        down_x = self.down_layer(x)
        hidden = self.gelu(down_x)
        up_x = self.up_layer(hidden) + x

        up_x = mask * up_x + (1 - mask) * x
        return up_x


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_layers = config.n_layers
        self.layer = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: Optional[bool] = None,
    ) -> Union[BaseModelOutput, Tuple[torch.Tensor, ...]]:  # docstyle-ignore
        """
        Parameters:
            x: torch.tensor(bs, seq_length, dim) Input sequence embedded.
            attn_mask: torch.tensor(bs, seq_length) Attention mask on the sequence.
        Returns:
            hidden_state: torch.tensor(bs, seq_length, dim) Sequence of hidden states in the last (top)
            layer all_hidden_states: Tuple[torch.tensor(bs, seq_length, dim)]
                Tuple of length n_layers with the hidden states from each layer.
                Optional: only if output_hidden_states=True
            all_attentions: Tuple[torch.tensor(bs, n_heads, seq_length, seq_length)]
                Tuple of length n_layers with the attention weights from each layer
                Optional: only if output_attentions=True
        """
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_state = x
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)

            layer_outputs = layer_module(
                x=hidden_state, attn_mask=attn_mask, head_mask=head_mask[i], output_attentions=output_attentions
            )
            hidden_state = layer_outputs[-1]

            if output_attentions:
                assert len(layer_outputs) == 2
                attentions = layer_outputs[0]
                all_attentions = all_attentions + (attentions,)
            else:
                assert len(layer_outputs) == 1

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)

        if not return_dict:
            return tuple(v for v in [hidden_state, all_hidden_states, all_attentions] if v is not None)

        return BaseModelOutput(
            last_hidden_state=hidden_state, hidden_states=all_hidden_states, attentions=all_attentions
        )


class BaseAdapterTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_layers = config.n_layers
        self.layer = nn.ModuleList([BaseAdapterTransformerBlock(config) for _ in range(config.n_layers)])
        self.current_idx = 0

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: Optional[bool] = None,
    ) -> Union[BaseModelOutput, Tuple[torch.Tensor, ...]]:  # docstyle-ignore
        """
        Parameters:
            x: torch.tensor(bs, seq_length, dim) Input sequence embedded.
            attn_mask: torch.tensor(bs, seq_length) Attention mask on the sequence.
        Returns:
            hidden_state: torch.tensor(bs, seq_length, dim) Sequence of hidden states in the last (top)
            layer all_hidden_states: Tuple[torch.tensor(bs, seq_length, dim)]
                Tuple of length n_layers with the hidden states from each layer.
                Optional: only if output_hidden_states=True
            all_attentions: Tuple[torch.tensor(bs, n_heads, seq_length, seq_length)]
                Tuple of length n_layers with the attention weights from each layer
                Optional: only if output_attentions=True
        """
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_state = x
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)

            layer_outputs = layer_module(
                x=hidden_state, attn_mask=attn_mask, head_mask=head_mask[i], output_attentions=output_attentions, adapter_idx=self.current_idx
            )
            hidden_state = layer_outputs[-1]

            if output_attentions:
                assert len(layer_outputs) == 2
                attentions = layer_outputs[0]
                all_attentions = all_attentions + (attentions,)
            else:
                assert len(layer_outputs) == 1

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)

        if not return_dict:
            return tuple(v for v in [hidden_state, all_hidden_states, all_attentions] if v is not None)

        return BaseModelOutput(
            last_hidden_state=hidden_state, hidden_states=all_hidden_states, attentions=all_attentions
        )

class BaseAdapterBertEncoder(nn.Module):
    def __init__(self, config, rank=32):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BaseAdapterBertLayer(config, rank) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class BaseAdapterBertLayer(nn.Module):
    def __init__(self, config, rank=16):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAdapterAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = BertAttention(config, position_embedding_type="absolute")
        self.intermediate = BertIntermediate(config)
        self.output = BertAdapterOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)        
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertAdapterAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = BertAttention(config, position_embedding_type=position_embedding_type)
        self.output = BertAdapterSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs



class BertAdapterOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.adapter = BaseAdapter(config.hidden_size, 16)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        hidden_states = self.adapter(hidden_states)

        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class MixClassifier(nn.Module):
    def __init__(self, config, num_labels=4):
        super().__init__()
        self.n_experts = 5
        self.mix_linear = nn.ModuleList([nn.Linear(config.hidden_size, num_labels) for _ in range(self.n_experts)])
    
    def forward(self, x):
        # hidden_states = self.adapter(hidden_states)
        rand_idx = np.random.choice(self.n_experts, 1)[0]
        return self.mix_linear[rand_idx](x)

class BertMixAdapterOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.n_experts = 3
        self.adapter = nn.ModuleList([BaseAdapter(config.hidden_size, 32) for _ in range(self.n_experts)])

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # hidden_states = self.adapter(hidden_states)
        rand_idx = np.random.choice(self.n_experts, 1)[0]
        hidden_states = self.adapter[rand_idx](hidden_states)

        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertMixAdapterSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.n_experts = 3
        self.adapter = nn.ModuleList([BaseAdapter(config.hidden_size, 32) for _ in range(self.n_experts)])

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        rand_idx = np.random.choice(self.n_experts, 1)[0]
        hidden_states = self.adapter[rand_idx](hidden_states)
        
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states    

class FusionClassifier(nn.Module):
    def __init__(self, config, num_labels=4):
        super().__init__()
        self.n_experts = 5

        # self.mix_linear = nn.ModuleList([nn.Linear(config.hidden_size, num_labels) for _ in range(self.n_experts)])
        self.main_cls = nn.Linear(config.hidden_size, num_labels)
    
    def forward(self, x):
        pred = self.main_cls(x)
        return pred

class BertFusionAdapterOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.main_adapter = StochasticAdapter(config.hidden_size, 16)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        hidden_states = self.main_adapter(hidden_states)

        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertFusionAdapterSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # self.positive_adapter = BaseAdapter(config.hidden_size, 32)
        self.main_adapter = StochasticAdapter(config.hidden_size, 16)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = self.main_adapter(hidden_states)

        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states    

class BertAdapterSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.adapter = BaseAdapter(config.hidden_size, 16)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        hidden_states = self.adapter(hidden_states)
        
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    
class BertSelectOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.adapter = AdaMixAdapter(config.hidden_size, 16)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        hidden_states = self.adapter(hidden_states)

        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    

class BertSelectSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.adapter = AdaMixAdapter(config.hidden_size, 64)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        hidden_states = self.adapter(hidden_states)
        
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertRoutingLoRAEncoder(nn.Module):
    def __init__(self, config, prefix_len=10):
        super().__init__()
        self.config = config
        self.prefix_len = prefix_len
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.fixed_layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False
        
        self.routing_probs = nn.Parameter(torch.rand(32)) # TODO batchsize
        self.routing_probs.requires_grad = False
        self.start_routing = False
        self.UB = 0.85

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        prefix_mask = torch.zeros_like(attention_mask)
        attention_mask = torch.cat([prefix_mask[:,:,:,:self.prefix_len], attention_mask], dim=3)[:, :, :, :hidden_states.size(1)]

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if self.start_routing:
                mask = (torch.rand(hidden_states.size(0)).cuda() < self.UB * self.routing_probs.cuda()).float()
            else:
                mask = (torch.rand(hidden_states.size(0)).cuda() < self.UB).float()
                mask = (torch.ones_like(mask) * random.random() < self.UB).float()

            mask = mask.view(-1, 1, 1)

            # if output_hidden_states:
            #     all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
                
                fixed_layer_outputs = self.fixed_layer[i](
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            fixed_hidden_states = fixed_layer_outputs[0]
            
            hidden_states = mask * hidden_states + (1 - mask) * fixed_hidden_states
            
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
        

class BertRoutingBitFitEncoder(nn.Module):
    def __init__(self, config, prefix_len=10):
        super().__init__()
        self.config = config
        self.prefix_len = prefix_len
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.fixed_layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False
        
        self.routing_probs = nn.Parameter(torch.rand(32)) # TODO batchsize
        self.routing_probs.requires_grad = False
        self.start_routing = False
        self.inference_mode = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        prefix_mask = torch.zeros_like(attention_mask)
        attention_mask = torch.cat([prefix_mask[:,:,:,:self.prefix_len], attention_mask], dim=3)[:, :, :, :hidden_states.size(1)]

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if not self.inference_mode:
                if self.start_routing:
                    mask = (torch.rand(hidden_states.size(0)).cuda() < 0.9 * self.routing_probs.cuda()).float()
                else:
                    mask = (torch.rand(hidden_states.size(0)).cuda() < 0.9 ).float()
            else:
                mask = torch.ones(hidden_states.size(0)).cuda()
            mask = mask.view(-1, 1, 1)

            # if output_hidden_states:
            #     all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
                
                fixed_layer_outputs = self.fixed_layer[i](
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            fixed_hidden_states = fixed_layer_outputs[0]
            
            hidden_states = mask * hidden_states + (1 - mask) * fixed_hidden_states
            
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

class BertPrefixEncoder(nn.Module):
    def __init__(self, config, prefix_len=20):
        super().__init__()
        self.config = config
        self.prefix_len = prefix_len
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False
        
        # self.prefix_embedding = nn.Parameter(torch.randn(config.num_hidden_layers, prefix_len, config.hidden_size))
        self.prefix_embedding = nn.Embedding(config.num_hidden_layers, prefix_len * config.hidden_size)
        # self.prefix_embedding.requires_grad = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        prefix_mask = torch.zeros_like(attention_mask)
        attention_mask = torch.cat([prefix_mask[:,:,:,:self.prefix_len], attention_mask], dim=3)[:, :, :, :hidden_states.size(1)]

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):

            # Add prefix embeddings 
            # prefix = self.prefix_embedding[i].unsqueeze(0).repeat(len(hidden_states), 1, 1)
            prefix = self.prefix_embedding(torch.LongTensor([i]).to(hidden_states.device)).view(1, self.prefix_len, -1).repeat(len(hidden_states), 1, 1)

            prefix_hidden_states = torch.cat([hidden_states[:, 0].unsqueeze(1), prefix, hidden_states[:, 1:]], dim=1)[:, :hidden_states.size(1)]
            
            prefix_mask = torch.zeros_like(attention_mask)
            prefix_attention_mask = torch.cat([prefix_mask[:,:,:,:self.prefix_len], attention_mask], dim=3)[:, :, :, :hidden_states.size(1)]  
            
            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    prefix_hidden_states,
                    prefix_attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

            # reset hidden_states, attention_mask
            padding = torch.zeros_like(hidden_states)[:, :self.prefix_len]
            hidden_states = torch.cat([hidden_states[:, 0].unsqueeze(1), hidden_states[:, self.prefix_len+1:], padding], dim=1)
            # hidden_states = 

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )



class BertRoutingPrefixEncoder(nn.Module):
    def __init__(self, config, prefix_len=20):
        super().__init__()
        self.config = config
        self.prefix_len = prefix_len
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False
        
        # self.prefix_embedding = nn.Parameter(torch.randn(config.num_hidden_layers, self.prefix_len, config.hidden_size))
        # self.prefix_embedding.requires_grad = True
        self.prefix_embedding = nn.Embedding(config.num_hidden_layers, prefix_len * config.hidden_size)

        
        self.routing_probs = nn.Parameter(torch.rand(32)) # TODO batchsize
        self.routing_probs.requires_grad = False
        self.start_routing = False
        self.UB = 0.85

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # prefix_mask = torch.zeros_like(attention_mask)
        # attention_mask = torch.cat([prefix_mask[:,:,:,:self.prefix_embedding.size(1)], attention_mask], dim=3)[:, :, :, :hidden_states.size(1)]

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            
            if self.start_routing:
                mask = (torch.rand(hidden_states.size(0)).cuda() < self.UB * self.routing_probs.cuda()).float()
            else:
                mask = (torch.rand(hidden_states.size(0)).cuda() < self.UB).float()
                mask = (torch.ones_like(mask) * random.random() < self.UB).float()
                
            mask = mask.view(-1, 1, 1, 1)           
             
            prefix = self.prefix_embedding(torch.LongTensor([i]).to(mask.device)).view(1, self.prefix_len, -1).repeat(len(hidden_states), 1, 1)
            prefix_hidden_states = torch.cat([hidden_states[:, 0].unsqueeze(1), prefix, hidden_states[:, 1:]], dim=1)[:, :hidden_states.size(1)]
            
            prefix_mask = torch.zeros_like(attention_mask)
            prefix_attention_mask = torch.cat([attention_mask[:,:,:,0].unsqueeze(-1), prefix_mask[:,:,:,:self.prefix_len], attention_mask[:,:,:,1:]], dim=3)[:, :, :, :hidden_states.size(1)]  
            
            non_prefix_mask = torch.ones_like(attention_mask) * -1e34
            non_prefix_attention_mask = torch.cat([attention_mask[:,:,:,0].unsqueeze(-1), non_prefix_mask[:,:,:,:self.prefix_len], attention_mask[:,:,:,1:]], dim=3)[:, :, :, :hidden_states.size(1)]  
            
            # routing_hidden_states = mask * prefix_hidden_states + (1 - mask) * hidden_states
            routing_attention_mask = mask * prefix_attention_mask + (1-mask) * non_prefix_attention_mask
            # routing_attention_mask[:,:,:,0] = 0

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    prefix_hidden_states,
                    routing_attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

            # reset hidden_states, attention_mask
            padding = torch.zeros_like(hidden_states)[:, :self.prefix_len]
            hidden_states = torch.cat([hidden_states[:, 0].unsqueeze(1), hidden_states[:, self.prefix_len+1:], padding], dim=1)

            # if output_hidden_states:
            #     all_hidden_states = all_hidden_states + (hidden_states.detach().cpu(),)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )