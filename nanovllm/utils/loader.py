"""import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    weight_name_mapping = getattr(model, "weight_name_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # Apply direct name mapping (e.g. kv_a_proj_with_mqa -> kv_a_proj)
                mapped_name = weight_name
                for old_name, new_name in weight_name_mapping.items():
                    if old_name in mapped_name:
                        mapped_name = mapped_name.replace(old_name, new_name)

                for k in packed_modules_mapping:
                    if k in mapped_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = mapped_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(mapped_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))"""



import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    weight_name_mapping = getattr(model, "weight_name_mapping", {})

    skip_keywords = [
        ".mlp.",
        ".experts.",
        ".gate.",
        ".shared_experts.",
    ]

    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                if any(k in weight_name for k in skip_keywords):
                    continue

                mapped_name = weight_name
                for old_name, new_name in weight_name_mapping.items():
                    if old_name in mapped_name:
                        mapped_name = mapped_name.replace(old_name, new_name)

                try:
                    for k in packed_modules_mapping:
                        if k in mapped_name:
                            v, shard_id = packed_modules_mapping[k]
                            param_name = mapped_name.replace(k, v)
                            param = model.get_parameter(param_name)
                            weight_loader = getattr(param, "weight_loader")
                            weight_loader(param, f.get_tensor(weight_name), shard_id)
                            break
                    else:
                        param = model.get_parameter(mapped_name)
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, f.get_tensor(weight_name))
                except (AttributeError, KeyError):
                    continue