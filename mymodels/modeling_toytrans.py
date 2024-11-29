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
from configuration_toytrans import ToyTransConfig
import logging

from transformers.modeling_utils import PreTrainedModel, SequenceSummary
import torch.nn.functional as F

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = ""
_CONFIG_FOR_DOC = "GPT2Config"








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

        self.embed_dim = config.embed_dim
        #self.seq_len = seq_len
        # Learnable parameters for query, key, and value

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)

        self.query = nn.Linear(self.embed_dim, self.embed_dim)
        self.key = nn.Linear(self.embed_dim, self.embed_dim)
        self.value = nn.Linear(self.embed_dim, self.embed_dim)

        # Feedforward layer
        self.fc1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.fc2 = nn.Linear(self.embed_dim, self.embed_dim)

        # Normalization
        #self.layer_norm1 = nn.LayerNorm(self.embed_dim)
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

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kargs,
    ) :

        # Compute Q, K, V
        embed = self.wte(input_ids)
        Q = self.query(embed)
        K = self.key(embed)
        V = self.value(embed)

        # Apply attention
        attention_output = self.attention(Q, K, V)

        # Residual connection and normalization
        #x = self.layer_norm1(attention_output)

        # Feedforward layer
        x = F.relu(self.fc1(x))
        ff_output = F.relu(self.fc2(x))

        # Second residual connection and normalization
        #output = self.layer_norm2(ff_output)

        if output_hidden_states:
            return (ff_output, (attention_output,embed), )

        else:
            return ff_output,


class ToyTransLMHeadModel(ToyTransModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = ToyTransModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

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
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kargs
    ):

        transformer_outputs = self.transformer(
            input_ids,
        )
        hidden_states = transformer_outputs

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

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

        if output_hidden_states:
            output = (lm_logits,) + transformer_outputs[1:]
        else:
            output = (lm_logits,)
        return ((loss,) + lm_logits) if loss is not None else output




