# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

import torch
from importlib import import_module
from zoedepth.models.depth_model import DepthModel
from zoedepth.models.model_io import load_wts
from zoedepth.models.base_models.midas import MidasCore
from zoedepth.models.zoedepth.zoedepth_v1 import ZoeDepth

from thirdparty.MiDaS.midas.dpt_depth import DPTDepthModel
# from thirdparty.MiDaS.models.midas_net import MidasNet_small

def build_model(config) -> DepthModel:
    """Builds a model from a config. The model is specified by the model name and version in the config. The model is then constructed using the build_from_config function of the model interface.
    This function should be used to construct models for training and evaluation.

    Args:
        config (dict): Config dict. Config is constructed in utils/config.py. Each model has its own config file(s) saved in its root model folder.

    Returns:
        torch.nn.Module: Model corresponding to name and version as specified in config
    """
    module_name = f"zoedepth.models.{config.model}"
    # print(config.use_pretrained_midas)
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as e:
        # print the original error message
        print(e)
        raise ValueError(
            f"Model {config.model} not found. Refer above error for details.") from e
    try:
        get_version = getattr(module, "get_version")
    except AttributeError as e:
        raise ValueError(
            f"Model {config.model} has no get_version function.") from e
    return get_version(config.version_name).build_from_config(config)

def build_model_from_pretrained(config, ckpt_path) -> DepthModel:
    """Builds a model from a config. The model is specified by the model name and version in the config. The model is then constructed using the build_from_config function of the model interface.
    This function should be used to construct models for training and evaluation.

    Args:
        config (dict): Config dict. Config is constructed in utils/config.py. Each model has its own config file(s) saved in its root model folder.

    Returns:
        torch.nn.Module: Model corresponding to name and version as specified in config
    """

    # 加载ZoeDepth模型权重
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    midas_weights_direct = {}
    for key, value in checkpoint['model'].items():
        if key.startswith('core.'):
            midas_key = key[5:]  # 去掉'core.'前缀
            midas_weights_direct[midas_key] = value

    # 加载MiDaS模型权重    
    midas = DPTDepthModel(
            path=None,
            backbone="beitl16_384",
            non_negative=True,
        )
    midas_core = MidasCore(midas)
    midas_core.set_output_channels(config.midas_model_type)
    midas_core.load_state_dict(midas_weights_direct)
    print('加载MiDaS模型权重成功')
    model = ZoeDepth(midas_core, **config)
    model = load_wts(model, ckpt_path)
    return model