# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch OpenAI GPT-2 model."""

import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from packaging import version
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from .configuration_toytrans import ToyTransConfig
import logging

from transformers.modeling_utils import PreTrainedModel, SequenceSummary
from transformers.generation import GenerationMixin
import torch.nn.functional as F




from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions, BaseModelOutputWithPastAndCrossAttentions


class ToyTransPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ToyTransConfig
    base_model_prefix = "transformer"
    is_parallelizable = True
    supports_gradient_checkpointing = True
    _no_split_modules = ["GPT2Block"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name == "c_proj.weight":
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                p.data.normal_(mean=0.0, std=(self.config.initializer_range / math.sqrt(2 * self.config.n_layer)))


class ToyTransModel(ToyTransPreTrainedModel):
    _supports_param_buffer_assignment = False

    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.hidden_size

        self.embed_dim = config.n_embd
        #self.seq_len = seq_len
        # Learnable parameters for query, key, and value

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)

        self.query = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.key = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.value = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        # Feedforward layer
        self.fc1 = nn.Linear(self.embed_dim, self.embed_dim*10)
        self.fc2 = nn.Linear(self.embed_dim*10, self.embed_dim*10)
        self.fc3 = nn.Linear(self.embed_dim*10, self.embed_dim)

        # Normalization
        #self.layer_norm0 = nn.LayerNorm(self.embed_dim)
        #self.layer_norm1 = nn.LayerNorm(self.embed_dim*10)
        #self.layer_norm2 = nn.LayerNorm(self.embed_dim)

        # Initialize weights and apply final processing
        self.post_init()


    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    def attention(self, Q, K, V):
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) #/ (self.embed_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)

        # Weighted sum of values
        output = torch.matmul(attention_weights, V)
        return output

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kargs,
    ) :

        presents = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # Compute Q, K, V
        embed = self.wte(input_ids)
        Q = self.query(embed)
        K = self.key(embed)
        V = self.value(embed)

        # Apply attention
        attention_output = self.attention(Q, K, V)
        #print(f"attention_output:{attention_output}")


        # Residual connection and normalization
        #attention_output = self.layer_norm0(attention_output)

        # Feedforward layer
        ff1_output = F.relu(self.fc1(attention_output))

        #ff1_output = self.layer_norm1(ff1_output)

        ff2_output = F.relu(self.fc2(ff1_output))


        ff3_output = F.relu(self.fc3(ff2_output))

        # Second residual connection and normalization
        #ff2_output = self.layer_norm2(ff2_output)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (ff3_output, ff2_output, ff1_output, attention_output, embed,)

        if not return_dict:
            return tuple(
                v
                for v in [ff3_output, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=ff3_output,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )



class ToyTransLMHeadModel(ToyTransPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = ToyTransModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head_backup = None

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.tokenizer_offset = 0
        self.debug = False

        # Initialize weights and apply final processing
        self.post_init()


    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            return_dict: Optional[bool] = None,
            **kargs
    ):
        input_ids = input_ids - self.tokenizer_offset
        input_ids[input_ids<0] = 0
        #print(input_ids)
        transformer_outputs = self.transformer(
            input_ids,
            labels=labels,
            return_dict=return_dict,
            output_hidden_states=True,
            **kargs,
        )
        hidden_states = transformer_outputs

        # Set device for model parallelism
        #if self.model_parallel:
        #    torch.cuda.set_device(self.transformer.first_device)
        #    hidden_states = hidden_states.to(self.lm_head.weight.device)
        attention_result = ""
        embedding_result = ""
        shift_labels = ""

        if isinstance(hidden_states, tuple):
            lm_head_input = hidden_states[0]
        elif isinstance(hidden_states,BaseModelOutputWithPastAndCrossAttentions):
            lm_head_input = hidden_states.last_hidden_state
        else:
            lm_head_input = hidden_states
        lm_logits = self.lm_head(lm_head_input)
        #if self.tokenizer_offset > 0:
        if self.lm_head_backup is not None:
            lm_logits_backup = self.lm_head_backup(lm_head_input)
            #lm_logits_backup = torch.zeros(lm_logits.shape[0],lm_logits.shape[1],self.tokenizer_offset).to(dtype=lm_logits.dtype,device=lm_logits.device)
            lm_logits = torch.cat([lm_logits_backup, lm_logits],2)
        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if self.debug:
            if isinstance(transformer_outputs,dict):
                attention_result = transformer_outputs["hidden_states"][-2]
                embedding_result = transformer_outputs["hidden_states"][-1]
            elif isinstance(transformer_outputs, tuple):
                attention_result = transformer_outputs[1][-2]
                embedding_result = transformer_outputs[1][-1]
            print(f"input_labels:{input_ids}")
            print(f"attention_result:{attention_result}")
            print(f"embedding_result:{embedding_result}")
            print(f"shift_labels:{shift_labels}")
        #print(torch.sum(self.transformer.wte.weight))
        #print(torch.sum(self.transformer.query.weight))
        #print(torch.sum(self.transformer.key.weight))
        #print(torch.sum(self.transformer.value.weight))
        #print(torch.sum(self.lm_head_backup.weight))
        #if output_hidden_states:
        #    output = (lm_logits,) + transformer_outputs[1:]
        #else:
        #    output = lm_logits
        #return ((loss,) + lm_logits) if loss is not None else output
        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )


    @staticmethod
    def _reorder_cache(
            past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past_key_values
        )

    def resize_token_embeddings_by_tokenizer(self, tokenizer, reinitialize=True):
        self.valid_tokens = {}
        for ikey, ival in self.config.embed_map.items():
            token_encoded = tokenizer.encode(ikey)
            if len(token_encoded) == 1:
                self.valid_tokens[token_encoded[0]] = ival
        self.tokenizer_offset = min(self.valid_tokens.keys())
        if reinitialize:
            self.resize_token_embeddings(max(self.valid_tokens.keys()) - self.tokenizer_offset + 1)
        self.lm_head_backup = nn.Linear(self.transformer.embed_dim, self.tokenizer_offset, bias=False)
        with torch.no_grad():
            if reinitialize:
                self.transformer.wte.weight[:] = torch.zeros(self.transformer.wte.weight.shape).to(dtype=self.transformer.wte.weight.dtype)
                self.transformer.query.weight[:] = torch.ones(self.transformer.query.weight.shape).to(dtype=self.transformer.query.weight.dtype)
                self.transformer.key.weight[:] = torch.ones(self.transformer.key.weight.shape).to(dtype=self.transformer.key.weight.dtype)
                self.transformer.value.weight[:] = torch.eye(self.transformer.value.weight.shape[0]).to(dtype=self.transformer.key.weight.dtype)
                self.transformer.wte.weight.requires_grad = False
                self.transformer.query.weight.requires_grad = False
                self.transformer.key.weight.requires_grad = False
                self.transformer.value.weight.requires_grad = False
            self.lm_head_backup.weight[:] = torch.zeros(self.lm_head_backup.weight.shape).to(self.lm_head_backup.weight.dtype)
            self.lm_head_backup.weight.requires_grad = False
            for tkey,tval in self.valid_tokens.items():
                self.transformer.wte.weight[tkey-self.tokenizer_offset,tval] = 1



