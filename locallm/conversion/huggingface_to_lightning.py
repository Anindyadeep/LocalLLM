# Download LLMs from Hugging Face and then use this to convert to PyTorch Lightning

import gc
import sys
import json
import contextlib
from pathlib import Path
from functools import partial
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
from locallm.conversion.base import BaseConverter
from locallm.utils import NotYetLoadedTensor, incremental_save, lazy_load

# Acknowledgement: Lightning AI LitGPT
# https://github.com/Lightning-AI/lit-gpt/blob/main/scripts/convert_hf_checkpoint.py

class HuggingFaceToLightningConverter(BaseConverter):
    def _layer_template(layer_name: str, idx: int) -> Tuple[str, int]:
        split = layer_name.split(".")
        number = int(split[idx])
        split[idx] = "{}"
        from_name = ".".join(split)
        return from_name, number

    def load_param(param: Union[torch.Tensor, NotYetLoadedTensor], name: str, dtype: Optional[torch.dtype]) -> torch.Tensor:
        if hasattr(param, "_load_tensor"):
            # support tensors loaded via `lazy_load()`
            print(f"Loading {name!r} into RAM")
            param = param._load_tensor()
        if dtype is not None and type(dtype) is not NotYetLoadedTensor and dtype != param.dtype:
            print(f"Converting {name!r} from {param.dtype} to {dtype}")
            param = param.to(dtype)
        return param

    def copy_weights_falcon(
            self, 
            size: Literal["7b", "40b"],
            state_dict: Dict[str, torch.Tensor],
            hf_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
            saver: Optional[incremental_save] = None,
            dtype: Optional[torch.dtype] = None
        ) -> None:
        weight_map = {
            "transformer.word_embeddings.weight": "transformer.wte.weight",
            "transformer.h.{}.self_attention.query_key_value.weight": "transformer.h.{}.attn.attn.weight",
            "transformer.h.{}.self_attention.dense.weight": "transformer.h.{}.attn.proj.weight",
            "transformer.h.{}.mlp.dense_h_to_4h.weight": "transformer.h.{}.mlp.fc.weight",
            "transformer.h.{}.mlp.dense_4h_to_h.weight": "transformer.h.{}.mlp.proj.weight",
            "transformer.ln_f.bias": "transformer.ln_f.bias",
            "transformer.ln_f.weight": "transformer.ln_f.weight",
            "lm_head.weight": "lm_head.weight",
        }

        if size == "7b":
            weight_map.update(
                {
                    "transformer.h.{}.input_layernorm.bias": "transformer.h.{}.norm_1.bias",
                    "transformer.h.{}.input_layernorm.weight": "transformer.h.{}.norm_1.weight",
                }
            )
        elif size == "40b":
            weight_map.update(
                {
                    "transformer.h.{}.ln_attn.bias": "transformer.h.{}.norm_1.bias",
                    "transformer.h.{}.ln_attn.weight": "transformer.h.{}.norm_1.weight",
                    "transformer.h.{}.ln_mlp.bias": "transformer.h.{}.norm_2.bias",
                    "transformer.h.{}.ln_mlp.weight": "transformer.h.{}.norm_2.weight",
                }
            )
        else:
            raise NotImplementedError
        
        for name, param in hf_weights.items():
            if "transformer.h" in name:
                from_name, number = self.layer_template(name, 2)
                to_name = weight_map[from_name].format(number)
            else:
                to_name = weight_map[name]
            param = self.load_param(param, name, dtype)
            if saver is not None:
                param = saver.store_early(param)
            state_dict[to_name] = param
    
    def copy_weights_gptneox(
            self,
            state_dict: Dict[str, torch.Tensor],
            hf_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
            saver: Optional[incremental_save] = None,
            dtype: Optional[torch.dtype] = None,
        ) -> None:
        weight_map = {
            "gpt_neox.embed_in.weight": "transformer.wte.weight",
            "gpt_neox.layers.{}.input_layernorm.bias": "transformer.h.{}.norm_1.bias",
            "gpt_neox.layers.{}.input_layernorm.weight": "transformer.h.{}.norm_1.weight",
            "gpt_neox.layers.{}.attention.query_key_value.bias": "transformer.h.{}.attn.attn.bias",
            "gpt_neox.layers.{}.attention.query_key_value.weight": "transformer.h.{}.attn.attn.weight",
            "gpt_neox.layers.{}.attention.dense.bias": "transformer.h.{}.attn.proj.bias",
            "gpt_neox.layers.{}.attention.dense.weight": "transformer.h.{}.attn.proj.weight",
            "gpt_neox.layers.{}.attention.rotary_emb.inv_freq": None,
            "gpt_neox.layers.{}.attention.bias": None,
            "gpt_neox.layers.{}.attention.masked_bias": None,
            "gpt_neox.layers.{}.post_attention_layernorm.bias": "transformer.h.{}.norm_2.bias",
            "gpt_neox.layers.{}.post_attention_layernorm.weight": "transformer.h.{}.norm_2.weight",
            "gpt_neox.layers.{}.mlp.dense_h_to_4h.bias": "transformer.h.{}.mlp.fc.bias",
            "gpt_neox.layers.{}.mlp.dense_h_to_4h.weight": "transformer.h.{}.mlp.fc.weight",
            "gpt_neox.layers.{}.mlp.dense_4h_to_h.bias": "transformer.h.{}.mlp.proj.bias",
            "gpt_neox.layers.{}.mlp.dense_4h_to_h.weight": "transformer.h.{}.mlp.proj.weight",
            "gpt_neox.final_layer_norm.bias": "transformer.ln_f.bias",
            "gpt_neox.final_layer_norm.weight": "transformer.ln_f.weight",
            "embed_out.weight": "lm_head.weight",
        }

        for name, param in hf_weights.items():
            if "gpt_neox.layers" in name:
                from_name, number = self.layer_template(name, 2)
                to_name = weight_map[from_name]
                if to_name is None:
                    continue
                to_name = to_name.format(number)
            else:
                to_name = weight_map[name]
            param = self.countload_param(param, name, dtype)
            if saver is not None:
                param = saver.store_early(param)
            state_dict[to_name] = param
    
    def copy_weights_llama(
            self,
            config: Config,
            qkv_weights: Dict[int, List[Optional[NotYetLoadedTensor]]],
            state_dict: Dict[str, torch.Tensor],
            hf_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
            saver: Optional[incremental_save] = None,
            dtype: Optional[torch.dtype] = None,
        ) -> None:
        weight_map = {
            "model.embed_tokens.weight": "transformer.wte.weight",
            "model.layers.{}.input_layernorm.weight": "transformer.h.{}.norm_1.weight",
            "model.layers.{}.self_attn.q_proj.weight": None,
            "model.layers.{}.self_attn.k_proj.weight": None,
            "model.layers.{}.self_attn.v_proj.weight": None,
            "model.layers.{}.self_attn.o_proj.weight": "transformer.h.{}.attn.proj.weight",
            "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
            "model.layers.{}.post_attention_layernorm.weight": "transformer.h.{}.norm_2.weight",
            "model.layers.{}.mlp.gate_proj.weight": "transformer.h.{}.mlp.fc_1.weight",
            "model.layers.{}.mlp.up_proj.weight": "transformer.h.{}.mlp.fc_2.weight",
            "model.layers.{}.mlp.down_proj.weight": "transformer.h.{}.mlp.proj.weight",
            "model.norm.weight": "transformer.ln_f.weight",
            "lm_head.weight": "lm_head.weight",
        }

        for name, param in hf_weights.items():
            if "model.layers" in name:
                from_name, number = self.layer_template(name, 2)
                qkv = qkv_weights.setdefault(number, [None, None, None])

                if "q_proj" in name:
                    qkv[0] = param
                elif "k_proj" in name:
                    qkv[1] = param
                elif "v_proj" in name:
                    qkv[2] = param
                to_name = weight_map[from_name]
                if to_name is None:
                    continue
                to_name = to_name.format(number)
            else:
                to_name = weight_map[name]
            param = self.load_param(param, name, dtype)
            
            if saver is not None:
                param = saver.store_early(param)
            state_dict[to_name] = param
        
        for i, (q, k, v) in list(qkv_weights.items()):
            if q is None or k is None or v is None:
                continue
            q = self.load_param(q, f"layer {i} q", dtype)
            k = self.load_param(k, f"layer {i} k", dtype)
            v = self.load_param(v, f"layer {i} v", dtype)
        


    def layer_template(self, layer_name: str, idx: int) -> Tuple[str, int]:
        split = layer_name.split(".")
        number = int(split[idx])
        split[idx] = "{}"
        from_name = ".".join(split)
        return from_name, number


    def load_param(self, param: Union[torch.Tensor, NotYetLoadedTensor], name: str, dtype: Optional[torch.dtype]) -> torch.Tensor:
        if hasattr(param, "_load_tensor"):
            # support tensors loaded via `lazy_load()`
            print(f"Loading {name!r} into RAM")
            param = param._load_tensor()
        if dtype is not None and type(dtype) is not NotYetLoadedTensor and dtype != param.dtype:
            print(f"Converting {name!r} from {param.dtype} to {dtype}")
            param = param.to(dtype)
        return param