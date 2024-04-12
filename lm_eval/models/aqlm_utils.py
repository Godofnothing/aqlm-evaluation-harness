import os

import torch
import torch.nn as nn

MODEL_ERROR_MSG = "Unsupported model type {} - only 'llama', 'Yi', 'opt' and 'falcon' are supported"
FALCON_TYPES = ("falcon", "refinedweb", "refinedwebmodel")
LLAMA_LIKE = ("llama", "Yi", "mistral", "mixtral", "gemma", "cohere")


def get_layers(model):
    if model.config.model_type in LLAMA_LIKE:
        return model.model.layers
    elif model.config.model_type.lower() in FALCON_TYPES:
        return model.transformer.h
    elif model.config.model_type == "opt":
        return model.model.decoder.layers
    else:
        raise ValueError(MODEL_ERROR_MSG.format(model.config.model_type))
    

def load_linear_layers(layer, quant_layer, model):
    layer_ident = {}
    for submodule in layer.modules():
        for child_name, child_module in submodule.named_children():
            print(child_name, "child_name", layer_ident)
            if isinstance(child_module, (nn.Conv2d, nn.Linear)) or "norm" in child_name:
                if child_name in layer_ident:
                    layer_ident[child_name] += 1
                else:
                    layer_ident[child_name] = 1
                quant_count = 0
                print("Finding to dequantize ", child_name)
                for quant_submodule in quant_layer.modules():
                    for quant_child_name, quant_child_module in quant_submodule.named_children():
                        if quant_child_name == child_name:
                            quant_count += 1
                            if quant_count != layer_ident[child_name]:
                                continue
                            print(quant_child_name, quant_child_module)
                            if ("gate" in child_name.lower()) and ("mixtral" in model.config.model_type.lower()):
                                print("gate", child_name)
                                child_module.weight.data = quant_child_module.weight.data.to(
                                    child_module.weight.dtype
                                ).to(child_module.weight.device)
                                continue
                            if "norm" in child_name and not isinstance(child_module, (nn.Conv2d, nn.Linear)):
                                print("norm", child_name)
                                child_module.weight.data = quant_child_module.weight.data.to(
                                    child_module.weight.dtype
                                ).to(child_module.weight.device)
                            else:
                                print(child_name)
                                child_module.weight.data = (
                                    quant_child_module.quantized_weight()
                                    .data.to(child_module.weight.dtype)
                                    .to(child_module.weight.device)
                                )
                            # Bias is not taked into account
    return layer


def load_dequantized_model(model, load_path):
    """Load quantized model by dequantizing it"""
    layers = get_layers(model)
    for layer_index in range(len(layers)):
        print("layer", layer_index)
        layer = layers[layer_index]
        quant_layer = torch.load(os.path.join(load_path, str(layer_index) + ".pth"), map_location="cpu")
        layers[layer_index] = load_linear_layers(layer, quant_layer, model)
    model.load_state_dict(torch.load(os.path.join(load_path, "not_quantized_weights.pt")), strict=False)
    return model