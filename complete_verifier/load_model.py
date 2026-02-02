#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
##   Copyright (C) 2021-2025 The α,β-CROWN Team                        ##
##   Primary contacts: Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##                     Zhouxing Shi <zshi@cs.ucla.edu> (UCLA)          ##
##                     Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##    See CONTRIBUTORS for all author contacts and affiliations.       ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
import hashlib
import time
from typing import Optional, List
import os
import collections
import gzip
from ast import literal_eval
import torch
import numpy as np

import onnx2pytorch
import onnx
import onnxruntime as ort
import onnxoptimizer
from skl2onnx.helpers.onnx_helper import select_model_inputs_outputs
from skl2onnx.helpers.onnx_helper import save_onnx_model
from skl2onnx.helpers.onnx_helper import enumerate_model_node_outputs
from onnx_opt import compress_onnx

import warnings
import importlib
from functools import partial

import arguments

from model_defs import *
from utils import expand_path


def Customized(def_file, callable_name, *args, **kwargs):
    """Fully customized model or dataloader."""
    if def_file.endswith('.py'):
        # Use relatively path w.r.t. to the configuration file
        if arguments.Config['general']['root_path']:
            path = os.path.join(
                expand_path(arguments.Config['general']['root_path']), def_file)
        elif arguments.Config.file:
            path = os.path.join(os.path.dirname(arguments.Config.file), def_file)
        else:
            path = def_file
        spec = importlib.util.spec_from_file_location('customized', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        try:
            # Customized loaders should be in "custom" folder.
            module = importlib.import_module(f'custom.{def_file}')
        except ModuleNotFoundError:
            # If not found, try the current folder.
            module = importlib.import_module(f'{def_file}')
            warnings.warn(  # Old config files may refer to custom loaders in the root folder.
                    f'Customized loaders "{def_file}" should be inside the "custom" folder.')
    # Load model from a specified file.
    model_func = getattr(module, callable_name)
    customized_func = partial(model_func, *args, **kwargs)
    # We need to return a Callable which returns the model.
    return customized_func


def deep_update(d, u):
    """Update a dictionary based another dictionary, recursively.

    (https://stackoverflow.com/a/3233356).
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def unzip_and_optimize_onnx(path, onnx_optimization_flags: Optional[List[str]] = None):
    if onnx_optimization_flags is None:
        onnx_optimization_flags = []
    if len(onnx_optimization_flags) == 0:
        if path.endswith('.gz'):
            onnx_model = onnx.load(gzip.GzipFile(path))
        else:
            onnx_model = onnx.load(path)
        return onnx_model
    else:
        print(f'Onnx optimization with flags: {onnx_optimization_flags}')
        npath = path + '.optimized'
        if os.path.exists(npath):
            print(f'Found existed optimized onnx model at {npath}')
            return onnx.load(npath)
        else:
            print(f'Generate optimized onnx model to {npath}')
            if path.endswith('.gz'):
                onnx_model = onnx.load(gzip.GzipFile(path))
            else:
                onnx_model = onnx.load(path)
            return compress_onnx(onnx_model, path, npath, onnx_optimization_flags, debug=True)


def inference_onnx(path, input):
    # Workaround for onnx bug, see issue #150
    options = ort.SessionOptions()
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = 1
    sess = ort.InferenceSession(unzip_and_optimize_onnx(path).SerializeToString(),
                                sess_options=options)
    assert len(sess.get_inputs()) == len(sess.get_outputs()) == 1
    res = sess.run(None, {sess.get_inputs()[0].name: input})[0]
    return res


@torch.no_grad()
def load_model_onnx(path, quirks=None, x=None):
    start_time = time.time()
    onnx_optimization_flags = arguments.Config['model']['onnx_optimization_flags']

    if arguments.Config['model']['cache_onnx_conversion']:
        cached_onnx_suffix = ".cached"
        cached_onnx_filename = f'{path}{cached_onnx_suffix}'

        with open(path, "rb") as file:
            curfile_sha256 = hashlib.sha256(file.read()).hexdigest()

        if os.path.exists(cached_onnx_filename):
            print(f'Loading cached onnx model from {cached_onnx_filename}')
            read_error = False
            try:
                pytorch_model, onnx_shape, old_file_sha256 = torch.load(cached_onnx_filename)
            except (Exception, ValueError, EOFError):
                print("Cannot read cached onnx file. Regenerating...")
                read_error = True
            if not read_error:
                if curfile_sha256 == old_file_sha256:
                    end_time = time.time()
                    print(f'Cached converted model loaded in {end_time - start_time:.4f} seconds')
                    return pytorch_model, onnx_shape
                else:
                    print(f"{cached_onnx_filename} file sha256: {curfile_sha256} does not match the current onnx sha256: {old_file_sha256}. Regenerating...")
        else:
            print(f"{cached_onnx_filename} does not exist.")

    quirks = {} if quirks is None else quirks
    if arguments.Config['model']['onnx_quirks']:
        try:
            config_quirks = literal_eval(arguments.Config['model']['onnx_quirks'])
        except ValueError:
            print('ERROR: onnx_quirks '
                  f'{arguments.Config["model"]["onnx_quirks"]}'
                  'cannot be parsed!')
            raise
        assert isinstance(config_quirks, dict)
        deep_update(quirks, config_quirks)
    print(f'Loading onnx {path} with quirks {quirks}')

    onnx_model = unzip_and_optimize_onnx(path, onnx_optimization_flags)

    if arguments.Config["model"]["input_shape"] is None:
        # find the input shape from onnx_model generally
        # https://github.com/onnx/onnx/issues/2657
        input_all = [node.name for node in onnx_model.graph.input]
        input_initializer = [node.name for node in onnx_model.graph.initializer]
        net_feed_input = list(set(input_all) - set(input_initializer))
        net_feed_input = [node for node in onnx_model.graph.input if node.name in net_feed_input]

        if len(net_feed_input) != 1:
            # in some rare case, we use the following way to find input shape
            # but this is not always true (collins-rul-cnn)
            net_feed_input = [onnx_model.graph.input[0]]

        onnx_input_dims = net_feed_input[0].type.tensor_type.shape.dim
        onnx_shape = tuple(d.dim_value for d in onnx_input_dims)
    else:
        # User specify input_shape
        onnx_shape = arguments.Config['model']['input_shape']

    # remove batch information
    # for nn4sys pensieve parallel, the first dimension of the input size is not batch, do not remove
    if onnx_shape[0] <= 1:
        onnx_shape = onnx_shape[1:]

    try:
        pytorch_model = onnx2pytorch.ConvertModel(
            onnx_model, experimental=True, quirks=quirks)
    except TypeError as e:
        print('\n\nA possible onnx2pytorch version error!')
        print('If you see "unexpected keyword argument \'quirks\'", that indicates your onnx2pytorch version is incompatible.')
        print('Please uninstall onnx2pytorch in your python environment (e.g., run "pip uninstall onnx2pytorch"), and then reinstall using:\n')
        print('pip install git+https://github.com/Verified-Intelligence/onnx2pytorch@fe7281b9b6c8c28f61e72b8f3b0e3181067c7399\n\n')
        print('The error below may not a bug of alpha-beta-CROWN. See instructions above.')
        raise(e)
    pytorch_model.eval()
    pytorch_model.to(dtype=torch.get_default_dtype())

    conversion_check_result = True
    try:
        # check conversion correctness
        # FIXME dtype of dummy may not match the onnx model, which can cause runtime error
        if x is not None:
            x = x.reshape(1, *onnx_shape)
        else:
            x = torch.randn([1, *onnx_shape])
        output_pytorch = pytorch_model(x).numpy()
        try:
            if arguments.Config['model']['check_optimized']:
                output_onnx = inference_onnx(path+'.optimized', x.numpy())
            else:
                output_onnx = inference_onnx(path, x.numpy())
        except ort.capi.onnxruntime_pybind11_state.InvalidArgument:
            # ONNX model might have shape problems. Remove the batch dimension and try again.
            output_onnx = inference_onnx(path, x.numpy().squeeze(0))
        if 'remove_relu_in_last_layer' in onnx_optimization_flags:
            output_pytorch = output_pytorch.clip(min=0)
        conversion_check_result = np.allclose(
            output_pytorch, output_onnx, 1e-4, 1e-5)
    except:  # pylint: disable=broad-except
        warnings.warn('Not able to check model\'s conversion correctness')
        print('\n*************Error traceback*************')
        import traceback; print(traceback.format_exc())
        print('*****************************************\n')
    if not conversion_check_result:
        print('\n**************************')
        print('Model might not be converted correctly. Please check onnx conversion carefully.')
        print('Output by pytorch:', output_pytorch)
        print('Output by onnx:', output_onnx)
        diff = torch.tensor(output_pytorch - output_onnx).abs().reshape(-1)
        print('Max error:', diff.max())
        index = diff.argmax()
        print('Max error index:', diff.argmax())
        print(f'Output by pytorch at {index}: ',
              torch.tensor(output_pytorch).reshape(-1)[index])
        print(f'Output by onnx at {index}: ',
              torch.tensor(output_onnx).reshape(-1)[index])
        print('**************************\n')

        if arguments.Config["model"]["debug_onnx"]:
            debug_onnx(onnx_model, pytorch_model, x.numpy())

    # TODO merge into the unzip_and_optimize_onnx()
    if arguments.Config["model"]["flatten_final_output"]:
        pytorch_model = nn.Sequential(pytorch_model, nn.Flatten())

    if arguments.Config["model"]["cache_onnx_conversion"]:
        torch.save((pytorch_model, onnx_shape, curfile_sha256), cached_onnx_filename)

    end_time = time.time()
    print(f'Finished onnx model loading in {end_time - start_time:.4f} seconds')

    return pytorch_model, onnx_shape


def debug_onnx(onnx_model, pytorch_model, dummy_input):
    path_tmp = '/tmp/debug.onnx'

    output_onnx = {}
    for node in enumerate_model_node_outputs(onnx_model):
        print('Inferencing onnx node:', node)
        save_onnx_model(select_model_inputs_outputs(onnx_model, node), path_tmp)
        optimized_model = onnxoptimizer.optimize(
            onnx.load(path_tmp),
            ["extract_constant_to_initializer",
             "eliminate_unused_initializer"])
        sess = ort.InferenceSession(optimized_model.SerializeToString())
        output_onnx[node] = torch.tensor(sess.run(
            None, {sess.get_inputs()[0].name: dummy_input})[0])

    print('Inferencing the pytorch model')
    output_pytorch = pytorch_model(
        torch.tensor(dummy_input), return_all_nodes=True)

    for k in output_pytorch:
        if k == sess.get_inputs()[0].name:
            continue
        print(k, output_onnx[k].shape)
        close = torch.allclose(output_onnx[k], output_pytorch[k])
        print('  close?', close)
        if not close:
            print('  max error', (output_onnx[k] - output_pytorch[k]).abs().max())

    import pdb; pdb.set_trace()


# def load_model(weights_loaded=True):
#     """
#     Load the model architectures and weights
#     """

#     assert arguments.Config["model"]["name"] is None or arguments.Config["model"]["onnx_path"] is None, (
#         "Conflict detected! User should specify model path by either --model or --onnx_path! "
#         "The cannot be both specified.")

#     assert arguments.Config["model"]["name"] is not None or arguments.Config["model"]["onnx_path"] is not None, (
#         "No model is loaded, please set --model <modelname> for pytorch model or --onnx_path <filename> for onnx model.")

#     if arguments.Config['model']['name'] is not None:
#         # You can customize this function to load your own model based on model name.
#         try:
#             model_ori = eval(arguments.Config['model']['name'])()  # pylint: disable=eval-used
#         except Exception:  # pylint: disable=broad-except
#             print(f'Cannot load pytorch model definition "{arguments.Config["model"]["name"]}()". '
#                   f'"{arguments.Config["model"]["name"]}()" must be a callable that returns a torch.nn.Module object.')
#             import traceback
#             traceback.print_exc()
#             exit()
#         model_ori.eval()

#         if not weights_loaded:
#             return model_ori

#         if arguments.Config["model"]["path"] is not None:
#             # Load pytorch model
#             # You can customize this function to load your own model based on model name.
#             sd = torch.load(expand_path(arguments.Config["model"]["path"]), map_location=torch.device('cpu'))
#             if 'state_dict' in sd:
#                 sd = sd['state_dict']
#             if isinstance(sd, list):
#                 sd = sd[0]
#             if not isinstance(sd, dict):
#                 raise NotImplementedError("Unknown model format, please modify model loader yourself.")
#             try:
#                 model_ori.load_state_dict(sd)
#             except RuntimeError:
#                 print('Failed to load the model')
#                 print('Keys in the state_dict of model_ori:')
#                 print(list(model_ori.state_dict().keys()))
#                 print('Keys in the state_dict trying to load:')
#                 print(list(sd.keys()))
#                 raise

#     elif arguments.Config["model"]["onnx_path"] is not None:
#         # Load onnx model
#         model_ori, _ = load_model_onnx(expand_path(
#             arguments.Config["model"]["onnx_path"]))

#     else:
#         print("Warning: pretrained model path is not given!")

#     print(model_ori)
#     print('Parameters:')
#     for p in model_ori.named_parameters():
#         print(f'  {p[0]}: shape {p[1].shape}')

#     return model_ori
import torch
import torch.nn as nn
import arguments
from utils import expand_path  # Assuming expand_path is in the same file or imported
# Add other necessary imports from your existing file here (e.g., load_model_onnx)

# =============================================================================
#  VERIFICATION HELPERS (Added for Model Swapping)
# =============================================================================

class GroupSort(nn.Module):
    """
    Universal GroupSort (1-Lipschitz).
    
    - If input is 4D (N, C, H, W): Acts as a 1x1 Convolution (Spatial).
    - If input is 2D (N, C): Acts as a Linear layer (Dense).
    
    Implemented via Conv2d weights to ensure maximum compatibility with 
    Auto_LiRPA's bound propagation graph.
    """
    def __init__(self, channels, axis=1):
        super().__init__()
        # Auto_LiRPA verification usually targets the channel axis (1).
        if axis != 1:
            raise ValueError("GroupSort axis must be 1 (Channel).")
        
        if channels % 2 != 0:
            raise ValueError(f"Channels must be even, got {channels}")
            
        self.channels = channels
        
        # We use Conv2d(1x1) as the core engine. 
        # For 2D inputs, we simply unsqueeze dimensions to fit this engine.
        self.conv_diff = nn.Conv2d(channels, channels // 2, kernel_size=1, bias=False)
        self.conv_expand = nn.Conv2d(channels // 2, channels, kernel_size=1, bias=False)
        
        # Freeze weights
        self.conv_diff.weight.requires_grad = False
        self.conv_expand.weight.requires_grad = False
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            # 1. Diff Conv: Calculate (Even - Odd)
            self.conv_diff.weight.fill_(0)
            for k in range(self.channels // 2):
                self.conv_diff.weight[k, 2*k, 0, 0] = 1.0     # Even
                self.conv_diff.weight[k, 2*k + 1, 0, 0] = -1.0 # Odd
            
            # 2. Expand Conv: Apply corrections
            self.conv_expand.weight.fill_(0)
            for k in range(self.channels // 2):
                # If (Even > Odd), subtract difference from Even (swap)
                self.conv_expand.weight[2*k, k, 0, 0] = -1.0
                # If (Even > Odd), add difference to Odd (swap)
                self.conv_expand.weight[2*k + 1, k, 0, 0] = 1.0

    def forward(self, x):
        # 1. Detect Input Type
        is_2d = (x.dim() == 2)
        
        if is_2d:
            # Case: Linear Layer Output (N, C)
            # We reshape to (N, C, 1, 1) so it looks like an image pixel.
            # This is safe for verification because there is no spatial structure to lose.
            x_reshaped = x.view(*x.shape, 1, 1)
        else:
            # Case: Conv Layer Output (N, C, H, W)
            # Use as is.
            x_reshaped = x

        # 2. Sort Logic (Identical for both)
        diff = self.conv_diff(x_reshaped)
        activation = torch.relu(diff)
        correction = self.conv_expand(activation)
        out_reshaped = x_reshaped + correction
        
        # 3. Restore Output Shape
        if is_2d:
            # Reshape (N, C, 1, 1) back to (N, C)
            return out_reshaped.view(x.shape)
        else:
            return out_reshaped

def replace_groupsort(model, dummy_input):
    """
    Replaces GroupSort_General (Reshape-based) layers with GroupSort (Conv1x1-based).
    """
    # Dictionary to store (layer_name -> input_channels)
    layer_configs = {}
    hooks = []

    # 1. Define Hook to capture shapes
    def get_shape_hook(name):
        def hook(module, input, output):
            # Input is a tuple, take the first element
            shape = input[0].shape
            channels = shape[1] 
            layer_configs[name] = channels
        return hook

    # 2. Register hooks on all GroupSort layers
    # We check string name to avoid needing to import GroupSort_General class
    for name, module in model.named_modules():
        if "GroupSort" in str(type(module)):
            hooks.append(module.register_forward_hook(get_shape_hook(name)))

    # 3. Run Dummy Pass to trigger hooks
    training_state = model.training
    model.eval()
    with torch.no_grad():
        try:
            model(dummy_input)
        except Exception as e:
            print(f"Warning: Dummy pass in replace_groupsort failed: {e}")
            # Clean up hooks even if it fails
            for h in hooks: h.remove()
            return model

    # 4. Remove hooks
    for h in hooks:
        h.remove()
    model.train(training_state)

    # 5. Perform Replacement
    if not layer_configs:
        # No GroupSort layers found, return original model
        return model

    print(f"[replace_groupsort] Found {len(layer_configs)} GroupSort layers to swap.")
    
    for name, module in model.named_modules():
        for child_name, _ in module.named_children():
            full_child_name = f"{name}.{child_name}" if name else child_name
            
            if full_child_name in layer_configs:
                channels = layer_configs[full_child_name]
                axis = 1 
                
                # print(f"  -> Replacing {full_child_name} (Channels={channels})")
                new_layer = GroupSort(channels=channels, axis=axis)
                setattr(module, child_name, new_layer)

    return model

# =============================================================================
#  UPDATED LOAD_MODEL
# =============================================================================

def load_model(weights_loaded=True):
    """
    Load the model architectures and weights
    """

    assert arguments.Config["model"]["name"] is None or arguments.Config["model"]["onnx_path"] is None, (
        "Conflict detected! User should specify model path by either --model or --onnx_path! "
        "The cannot be both specified.")

    assert arguments.Config["model"]["name"] is not None or arguments.Config["model"]["onnx_path"] is not None, (
        "No model is loaded, please set --model <modelname> for pytorch model or --onnx_path <filename> for onnx model.")

    if arguments.Config['model']['name'] is not None:
        # You can customize this function to load your own model based on model name.
        try:
            model_ori = eval(arguments.Config['model']['name'])()  # pylint: disable=eval-used
        except Exception:  # pylint: disable=broad-except
            print(f'Cannot load pytorch model definition "{arguments.Config["model"]["name"]}()". '
                  f'"{arguments.Config["model"]["name"]}()" must be a callable that returns a torch.nn.Module object.')
            import traceback
            traceback.print_exc()
            exit()
        
        model_ori.eval()

        if not weights_loaded:
            return model_ori

        if arguments.Config["model"]["path"] is not None:
            # Load pytorch model
            print(f"Loading state dict from: {arguments.Config['model']['path']}")
            sd = torch.load(expand_path(arguments.Config["model"]["path"]), map_location=torch.device('cpu'))
            if 'state_dict' in sd:
                sd = sd['state_dict']
            if isinstance(sd, list):
                sd = sd[0]
            if not isinstance(sd, dict):
                raise NotImplementedError("Unknown model format, please modify model loader yourself.")
            try:
                model_ori.load_state_dict(sd)
            except RuntimeError:
                print('Failed to load the model')
                print('Keys in the state_dict of model_ori:')
                print(list(model_ori.state_dict().keys()))
                print('Keys in the state_dict trying to load:')
                print(list(sd.keys()))
                raise

            # =========================================================================
            # [CRITICAL FIX] SWAP ARCHITECTURE FOR VERIFICATION (GroupSort -> Conv1x1)
            # =========================================================================
            print("[load_model] 🔄 Applying Architecture Swap (GroupSort General -> Conv1x1)...")
            
            # 1. Generate a dummy input to infer shapes (required by replace_groupsort)
            # Try to fetch input shape from config, default to CIFAR-10 shape (3, 32, 32)
            model_name = arguments.Config["model"].get("name", "").lower()
           
            if "mnist" in model_name:
                # MNIST / Fashion-MNIST: Grayscale 28x28
                input_shape = (1, 1, 28, 28)
                print(f"[load_model] 📉 Detected MNIST/Grayscale model: {model_name}")
            else:
                # Default (CIFAR-10 / CIFAR-100 / TinyImageNet): RGB 32x32
                input_shape = (1, 3, 32, 32)
                print(f"[load_model] 📉 Detected CIFAR/RGB model: {model_name}")
            
            # 2. Create Dummy Input
            # We already defined full 4D shape (N, C, H, W), so we can pass it directly
            dummy_input = torch.zeros(*input_shape)
            
            # 2. Perform the swap
            try:
                model_ori = replace_groupsort(model_ori, dummy_input)
                print("[load_model] ✅ Swap successful. Model is now verifier-friendly.")
            except Exception as e:
                print(f"[load_model] ❌ Error during model swap: {e}")
                import traceback
                traceback.print_exc()
            # =========================================================================

    elif arguments.Config["model"]["onnx_path"] is not None:
        # Load onnx model
        model_ori, _ = load_model_onnx(expand_path(
            arguments.Config["model"]["onnx_path"]))

    else:
        print("Warning: pretrained model path is not given!")

    print(model_ori)
    print('Parameters:')
    for p in model_ori.named_parameters():
        print(f'  {p[0]}: shape {p[1].shape}')

    return model_ori