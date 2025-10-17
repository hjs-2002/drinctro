import torch
import torch.nn as nn
from torch.nn import init
import timm
from transformers import CLIPModel
import clip

try:
    from .f3net import F3Net
    from .resnet_gram import get_GramNet
except:
    from f3net import F3Net
    from resnet_gram import get_GramNet


# fc layer weight init
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')  # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)

    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


# 当in_channel != 3 时，初始化模型的第一个Conv的weight， 把之前的通道copy input_chaneel/3 次
def init_imagenet_weight(_conv_stem_weight, input_channel=3):
    for i in range(input_channel // 3):
        if i == 0:
            _conv_stem_weight_new = _conv_stem_weight
        else:
            _conv_stem_weight_new = torch.cat([_conv_stem_weight_new, _conv_stem_weight], axis=1)

    return torch.nn.Parameter(_conv_stem_weight_new)


# TODO 加载训练后的权值时需要设置strict=False
class CLIPVisual(nn.Module):
    def __init__(self, model_name, num_classes=2, freeze_extractor=False):
        super(CLIPVisual, self).__init__()
        model = CLIPModel.from_pretrained(model_name)
        self.visual_model = model.vision_model
        if freeze_extractor:
            self.freeze(self.visual_model)
        self.fc = nn.Linear(in_features=model.vision_embed_dim, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.visual_model(x)
        x = self.fc(x[1])

        return x

    # 冻结网络层
    @staticmethod
    def freeze(model):
        for param in model.parameters():
            param.requires_grad = False


class CLIPModelV2(nn.Module):
    # 定义不同 CLIP 模型对应的输出通道数
    CHANNELS = {
        "RN50": 1024,  # ResNet50 模型的输出通道数
        "ViT-B/32": 512,  # Vision Transformer Base 32 模型的输出通道数
        "ViT-L/14": 768  # Vision Transformer Large 14 模型的输出通道数
    }

    def __init__(self, name='clip-RN50', num_classes=2, freeze_extractor=False):
        """
        初始化 CLIPModelV2 类。

        :param name: 模型的名称，默认值为 'clip-RN50'。需要将名称中的特定字符替换为 CLIP 库支持的格式。
        :param num_classes: 分类的类别数量，默认值为 2。
        :param freeze_extractor: 是否冻结特征提取器的参数，默认为 False。
        """
        super(CLIPModelV2, self).__init__()
        # 对输入的模型名称进行处理，替换特定字符以匹配 CLIP 库支持的格式
        name = name.replace('clip-', '').replace('L-', 'L/').replace('B-', 'B/')
        # 加载 CLIP 模型和预处理函数，设备设置为 CPU。训练时预处理由 Dataset 类处理，此处预处理函数不会使用
        self.model, self.preprocess = clip.load(name, device="cpu")
        # 若 freeze_extractor 为 True，则冻结特征提取器的参数
        if freeze_extractor:
            self.freeze(self.model)
            print(f'Freezing the feature extractors!')

        # 定义全连接层，输入维度为对应 CLIP 模型的输出通道数，输出维度为分类的类别数量
        self.fc = nn.Linear(self.CHANNELS[name], num_classes)

    def forward(self, x, return_feature=False):
        """
        前向传播函数。

        :param x: 输入的图像张量。
        :param return_feature: 是否返回特征向量，默认为 False。
        :return: 若 return_feature 为 True，返回图像的特征向量；否则返回分类预测结果。
        """
        # 使用 CLIP 模型对输入图像进行编码，得到特征向量
        features = self.model.encode_image(x)
        if return_feature:
            return features
        # 若不返回特征向量，则将特征向量传入全连接层得到分类预测结果
        return self.fc(features)

    # 冻结网络层
    @staticmethod
    def freeze(model):
        """
        冻结给定模型的所有参数，使其在训练过程中不更新。

        :param model: 要冻结参数的模型。
        """
        for param in model.parameters():
            param.requires_grad = False


class ContrastiveModels(nn.Module):
    def __init__(self, model_name, num_classes=2, pretrained=True, embedding_size=1024,
                 freeze_extractor=False):
        """
        初始化 ContrastiveModels 类。
        该类用于创建对比学习模型，支持多种基础模型。
        :param model_name: 要使用的模型名称，用于指定创建的基础模型类型。
        :param num_classes: 分类的类别数量，默认为 2。
        :param pretrained: 是否使用预训练的模型权重，默认为 True。
        :param embedding_size: 嵌入层的维度大小，默认为 1024。
        :param freeze_extractor: 是否冻结特征提取器的参数，默认为 False。
        """
        # 调用父类 nn.Module 的构造函数
        super(ContrastiveModels, self).__init__()
        # 保存传入的模型名称
        self.model_name = model_name
        # 保存传入的嵌入层维度大小
        self.embedding_size = embedding_size
        # 调用 get_models 函数创建基础模型，将输出类别数设置为 embedding_size
        self.model = get_models(model_name=model_name, pretrained=pretrained, num_classes=embedding_size,
                                freeze_extractor=freeze_extractor)
        # 注释掉的代码，若基础模型有 default_cfg 属性，可保存该属性
        # self.default_cfg = self.model.default_cfg
        # 定义全连接层，将嵌入层输出映射到分类类别数
        self.fc = nn.Linear(embedding_size, num_classes)

    def forward(self, x, return_feature=False):
        feature = self.model(x)
        y_pred = self.fc(feature)
        if return_feature:
            return y_pred, feature

        return y_pred

    def extract_feature(self, x):
        feature = self.model(x)

        return feature


def get_efficientnet_ns(model_name='tf_efficientnet_b3_ns', pretrained=True, num_classes=2, start_down=True):
    """
     # Coefficients:   width,depth,res,dropout
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
        'efficientnet-b8': (2.2, 3.6, 672, 0.5),
        'efficientnet-l2': (4.3, 5.3, 800, 0.5),
    :param model_name:
    :param pretrained:
    :param num_classes:
    :return:
    """
    net = timm.create_model(model_name, pretrained=pretrained)
    if not start_down:
        net.conv_stem.stride = (1, 1)
    n_features = net.classifier.in_features
    net.classifier = nn.Linear(n_features, num_classes)

    return net


def get_swin_transformers(model_name='swin_base_patch4_window7_224', pretrained=True, num_classes=2):
    """
    :param model_name: swin_base_patch4_window12_384   swin_base_patch4_window7_224 swin_base_patch4_window7_224_in22k
    :param pretrained:
    :param num_classes:
    :return:
    """
    net = timm.create_model(model_name, pretrained=pretrained)
    n_features = net.head.in_features
    net.head = nn.Linear(n_features, num_classes)

    return net


def get_convnext(model_name='convnext_base_in22k', pretrained=True, num_classes=2, in_channel=3):
    """
    :param model_name: convnext_base_384_in22ft1k, convnext_base_in22k
    :param pretrained:
    :param num_classes:
    :return:
    """
    net = timm.create_model(model_name, pretrained=pretrained)
    n_features = net.head.fc.in_features
    net.head.fc = nn.Linear(n_features, num_classes)

    if in_channel != 3:
        first_conv_weight = net.stem[0].weight
        first_out_channels = net.stem[0].out_channels
        first_conv = nn.Conv2d(in_channel, first_out_channels, kernel_size=4, stride=4)
        first_conv.weight = init_imagenet_weight(first_conv_weight, input_channel=in_channel)
        net.stem[0] = first_conv

    return net


def get_resnet(model_name='resnet200d', pretrained=True, num_classes=2):
    """
    :param model_name: resnet200d, input_size=512, resnet50
    :param pretrained:
    :param num_classes:
    :return:
    """
    net = timm.create_model(model_name, pretrained=pretrained)
    n_features = net.fc.in_features
    net.fc = nn.Linear(n_features, num_classes)

    return net


def get_clip_visual_model(model_name="openai/clip-vit-base-patch32", num_classes=2, pretrained=True,
                          freeze_extractor=False):
    if 'openai/clip' in model_name:
        model = CLIPVisual(model_name=model_name, num_classes=num_classes)
    else:
        # 'clip-' + 'name', clip-RN50, clip-ViT-L/14
        model = CLIPModelV2(name=model_name, num_classes=num_classes, freeze_extractor=freeze_extractor)

    return model

vitl14={"embed_dim": 768,
    "vision_cfg": {
        "image_size": 224,
        "layers": 24,
        "width": 1024,
        "patch_size": 14
    },
    "text_cfg": {
        "context_length": 77,
        "vocab_size": 49408,
        "width": 768,
        "heads": 12,
        "layers": 12
    }}

def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == 'bf16':
        cast_dtype = torch.bfloat16
    elif precision == 'fp16':
        cast_dtype = torch.float16
    return cast_dtype
def get_models(model_name='tf_efficientnet_b3_ns', pretrained=True, num_classes=2,
               in_channel=3, freeze_extractor=False, embedding_size=None):
    """
    根据传入的模型名称和参数，创建并返回相应的深度学习模型实例。

    :param model_name: 模型的名称，用于指定要创建的模型类型，默认为 'tf_efficientnet_b3_ns'。
    :param pretrained: 是否使用预训练的模型权重，默认为 True。
    :param num_classes: 模型的输出类别数量，默认为 2。
    :param in_channel: 输入图像的通道数，默认为 3。
    :param freeze_extractor: 是否冻结特征提取器的参数，默认为 False。
    :param embedding_size: 嵌入层的维度大小，默认为 None。当该参数有效时，将创建对比学习模型。
    :return: 相应的深度学习模型实例。
    """

    cast_dtype = get_cast_dtype('fp32')
    # 检查 embedding_size 是否有效，若有效则创建对比学习模型
    # if embedding_size is not None and isinstance(embedding_size, int) and embedding_size > 0:
    #     model = ContrastiveModels(model_name, num_classes, pretrained, embedding_size, freeze_extractor)
    if embedding_size is not None and isinstance(embedding_size, int) and embedding_size < 0:
        model = ContrastiveModels(model_name, num_classes, pretrained, embedding_size, freeze_extractor)
        print('Using the Contrastive Model: {}'.format(model_name))
    # 若模型名称包含 'efficientnet'，则创建 EfficientNet 系列模型
    elif 'efficientnet' in model_name:
        model = get_efficientnet_ns(model_name, pretrained, num_classes)
    # 若模型名称包含 'convnext'，则创建 ConvNeXt 系列模型
    elif 'convnext' in model_name:
        model = get_convnext(model_name, pretrained, num_classes, in_channel=in_channel)
    # 若模型名称包含 'swin'，则创建 Swin Transformer 系列模型
    elif 'swin' in model_name:
        model = get_swin_transformers(model_name, pretrained, num_classes)
    # 若模型名称包含 'clip'，则创建 CLIP 视觉模型（输入尺寸必须为224）
    elif 'clip' in model_name:
        model = get_clip_visual_model(model_name, num_classes, freeze_extractor=freeze_extractor)  # 输入尺寸必须为224
    # 此处重复判断 'swin'，可考虑删除
    elif 'swin' in model_name:
        model = get_swin_transformers(model_name, pretrained=pretrained, num_classes=num_classes)
    # 若模型名称包含 'gram'，则创建 GramNet 模型
    elif 'gram' in model_name:  # gram_resnet18
        model = get_GramNet(model_name.replace('gram_', ''))
    # 若模型名称包含 'resnet'，则创建 ResNet 系列模型
    elif 'resnet' in model_name:
        model = get_resnet(model_name, pretrained, num_classes)
    # 若模型名称为 'f3net'，则创建 F3Net 模型
    elif model_name == 'f3net':
        model = F3Net(num_classes=num_classes, img_width=299, img_height=299, pretrained=pretrained)
    # 若模型名称不匹配上述任何条件，则抛出异常
    
    elif model_name == 'InCTRL':
        model = InCTRL(
    embed_dim=vitl14['embed_dim'],
    vision_cfg=vitl14['vision_cfg'],
    text_cfg=vitl14['text_cfg'],
    quick_gelu=False,
    cast_dtype=cast_dtype)
        print('Using the InCTRL Model: {}'.format(model_name))
        
    else:
        raise NotImplementedError(model_name)

    return model


if __name__ == '__main__':
    import time
    image_size = 224
    # model = get_models(model_name='clip-ViT-L-14', num_classes=2, pretrained=False,
    #                    embedding_size=512)  # clip-ViT-L-14
    # print(model)
    model = get_models(model_name='InCTRL', num_classes=2, pretrained=False,
                       embedding_size=512)
    # print(model.default_cfg)
    model = model.to(torch.device('cpu'))
    img = torch.randn(1, 3, image_size, image_size)  # your high resolution picture
    start = time.time()
    times = 1
    for _ in range(times):
        out = model(img)
        if isinstance(out, tuple):
            print([o.shape for o in out])
        else:
            print(out.shape)
    print((time.time()-start)/times)

    # from torchsummary import summary
    # input_s = (3, image_size, image_size)
    # print(summary(model, input_s, device='cpu'))
    pass



""" CLIP Model

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
from dataclasses import dataclass
import logging
import math
from typing import Optional, Tuple, Union
from collections import OrderedDict
import re
from sklearn.metrics import pairwise

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch import Tensor
import open_clip.utils.misc as misc
import argparse
from functools import partial
from open_clip.utils.env import checkpoint_pathmgr as pathmgr
from open_clip.hf_model import HFTextEncoder
# from .hf_model import HFTextEncoder
from open_clip.modified_resnet import ModifiedResNet
# from .modified_resnet import ModifiedResNet
from open_clip.timm_model import TimmModel
from open_clip.transformer import LayerNormFp32, LayerNorm, QuickGELU, Attention, VisionTransformer, TextTransformer, VisionTransformer_Mul
from open_clip.new_utils import to_2tuple

from open_clip.vp import (
    PadPrompter,
    RandomPatchPrompter,
    FixedPatchPrompter
)

from torch.autograd import Variable, grad

PROMPT_TYPES = {
    "padding": PadPrompter,
    "random_patch": RandomPatchPrompter,
    "fixed_patch": FixedPatchPrompter
}


@dataclass
class CLIPVisionCfg:
    """
    配置类，用于定义CLIP视觉模型的各种超参数和配置选项。

    参数说明：
        layers (Union[Tuple[int, int, int, int], int]): Transformer编码器的层数。可以是一个整数表示所有阶段使用相同层数，
            或者一个四元组分别指定每个阶段的层数，默认为12。
        width (int): Transformer模型的隐藏层维度（即嵌入维度），默认为768。
        head_width (int): 注意力机制中每个注意力头的维度，默认为64。
        mlp_ratio (float): MLP隐藏层维度与嵌入维度的比例，默认为4.0。
        patch_size (int): 图像分块的大小（以像素为单位），默认为16。
        image_size (Union[Tuple[int, int], int]): 输入图像的尺寸。可以是整数（表示方形图像）或二元组（高、宽），
            默认为224。

        ls_init_value (Optional[float]): LayerScale初始化值。如果为None则禁用LayerScale，默认为None。
        patch_dropout (float): 训练过程中丢弃patch的比例，取值范围[0, 1)。0表示不进行丢弃，推荐在0.5到0.75之间，
            默认为0.0。
        input_patchnorm (bool): 是否对每个patch应用双重归一化（Dual PatchNorm）。该设置仅作用于输入层归一化，
            因为原始CLIP ViT设计中已经包含了后置LayerNorm，默认为False。
        global_average_pool (bool): 是否使用全局平均池化代替CLS token来获取最终特征表示。
            参考论文：https://arxiv.org/abs/2205.01580，默认为False。
        attentional_pool (bool): 是否在最后一层嵌入层使用attentional pooler，默认为False。
        n_queries (int): Attentional pooler中的查询向量数量，默认为256。
        attn_pooler_heads (int): Attentional pooling使用的注意力头数，默认为8。
        output_tokens (bool): 是否输出token序列，默认为True。

        timm_model_name (str): 使用timm库中的预定义模型名称。若提供有效名称，则会覆盖layers、width、patch_size等配置项，
            默认为None。
        timm_model_pretrained (bool): 是否加载ImageNet预训练权重，默认为False。
        timm_pool (str): TIMM模型的特征池化方式，可选值包括'abs_attn'、'rot_attn'、'avg'或空字符串''，默认为'avg'。
        timm_proj (str): TIMM模型输出的线性投影类型，支持'linear'、'mlp'或空字符串''，默认为'linear'。
        timm_proj_bias (bool): 最终投影层是否启用偏置项，默认为False。
        timm_drop (float): 分类头的dropout比例，默认为0.0。
        timm_drop_path (Optional[float]): 主干网络的随机深度(drop path)比率，None表示不使用，默认为None。
    """
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224

    ls_init_value: Optional[float] = None  # layer scale initial value
    patch_dropout: float = 0.  # what fraction of patches to dropout during training (0 would mean disabled and no patches dropped) - 0.5 to 0.75 recommended in the paper for optimal results
    input_patchnorm: bool = False  # whether to use dual patchnorm - would only apply the input layernorm on each patch, as post-layernorm already exist in original clip vit design
    global_average_pool: bool = False  # whether to global average pool the last embedding layer, instead of using CLS token (https://arxiv.org/abs/2205.01580)
    attentional_pool: bool = False  # whether to use attentional pooler in the last embedding layer
    n_queries: int = 256  # n_queries for attentional pooler
    attn_pooler_heads: int = 8  # n heads for attentional_pooling
    output_tokens: bool = True

    timm_model_name: str = None  # a valid model name overrides layers, width, patch_size
    timm_model_pretrained: bool = False  # use (imagenet) pretrained weights for named model
    timm_pool: str = 'avg'  # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    timm_proj: str = 'linear'  # linear projection for timm model output ('linear', 'mlp', '')
    timm_proj_bias: bool = False  # enable bias final projection
    timm_drop: float = 0.  # head dropout
    timm_drop_path: Optional[float] = None  # backbone stochastic depth

@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    width: int = 512
    heads: int = 8
    layers: int = 12
    ls_init_value: Optional[float] = None  # layer scale initial value
    hf_model_name: str = None
    hf_tokenizer_name: str = None
    hf_model_pretrained: bool = True
    proj: str = 'mlp'
    pooler_type: str = 'mean_pooler'
    embed_cls: bool = False
    pad_id: int = 0
    output_tokens: bool = False

def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == 'bf16':
        cast_dtype = torch.bfloat16
    elif precision == 'fp16':
        cast_dtype = torch.float16
    return cast_dtype


def get_input_dtype(precision: str):
    input_dtype = None
    if precision in ('bf16', 'pure_bf16'):
        input_dtype = torch.bfloat16
    elif precision in ('fp16', 'pure_fp16'):
        input_dtype = torch.float16
    return input_dtype

state_level = {
               "normal":["{}", "flawless {}", "perfect {}", "unblemished {}",
                         "{} without flaw", "{} without defect", "{} without damage"],
                "anomaly":["damaged {}", "{} with flaw", "{} with defect", "{} with damage"]
}
template_level = [
                  "a cropped photo of the {}.",
                  "a cropped photo of a {}.",
                  "a close-up photo of a {}.",
                  "a close-up photo of the {}.",
                  "a bright photo of a {}.",
                  "a bright photo of the {}.",
                  "a dark photo of a {}.",
                  "a dark photo of the {}.",
                  "a jpeg corrupted photo of a {}.",
                  "a jpeg corrupted photo of the {}.",
                  "a blurry photo of the {}.",
                  "a blurry photo of a {}.",
                  "a photo of the {}.",
                  "a photo of a {}.",
                  "a photo of a small {}.",
                  "a photo of the small {}.",
                  "a photo of a large {}.",
                  "a photo of the large {}.",
                  "a photo of a {} for visual inspection.",
                  "a photo of the {} for visual inspection.",
                  "a photo of a {} for anomaly detection.",
                  "a photo of the {} for anomaly detection."
]

def get_texts(obj_name):
    """
    根据对象名称生成正常和异常状态的文本描述列表，用于异常检测任务。

    对于预定义的基本对象类别（如飞机、汽车等），直接生成简单的正常/异常描述文本；
    对于其他对象，则使用模板和状态描述组合生成更丰富的文本描述。

    参数:
        obj_name (str): 对象名称

    返回:
        tuple: 包含两个列表的元组
            - normal_texts (list): 正常状态的文本描述列表
            - anomaly_texts (list): 异常状态的文本描述列表
    """
    
    # l = ["airplane", "automobile", "bird",
    #      "cat", "deer", "dog", "frog", "horse", "ship", "truck", "animal"]
    l = [ 'real','fake','ai generated','original','reconstructed','human face', 'cat', 'dog', 'bird', 'car', 'bus', 'truck', 'airplane', 'flower', 'tree', 'fruit', 'vegetable', 'fish', 'insect', 'mammal', 'reptile', 'amphibian', 'furniture', 'appliance', 'electronic device', 'clothing', 'accessory', 'tool', 'utensil', 'toy']
    if obj_name in l:
        # 对于基本对象类别，使用简单的文本描述
        reconstructed_texts = []
        original_texts = []
        reconstructed = "a reconstructed photo of " + obj_name + " for AIGC detection."
        reconstructed_texts.append(reconstructed)
        original = "a original photo of " + obj_name + " for AIGC detection."
        original_texts.append(original)
    else:
        # 对于其他对象，使用模板和状态描述组合生成丰富的文本描述
        normal_states = [s.format(obj_name) for s in state_level["normal"]]
        anomaly_states = [s.format(obj_name) for s in state_level["anomaly"]]

        normal_texts = [t.format(state) for state in normal_states for t in template_level]
        anomaly_texts = [t.format(state) for state in anomaly_states for t in template_level]

    return reconstructed_texts, original_texts


def _build_vision_tower(
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None
):
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
    # memory efficient in recent PyTorch releases (>= 1.10).
    # NOTE: timm models always use native GELU regardless of quick_gelu flag.
    act_layer = QuickGELU if quick_gelu else nn.GELU

    if vision_cfg.timm_model_name:
        visual = TimmModel(
            vision_cfg.timm_model_name,
            pretrained=vision_cfg.timm_model_pretrained,
            pool=vision_cfg.timm_pool,
            proj=vision_cfg.timm_proj,
            proj_bias=vision_cfg.timm_proj_bias,
            drop=vision_cfg.timm_drop,
            drop_path=vision_cfg.timm_drop_path,
            patch_drop=vision_cfg.patch_dropout if vision_cfg.patch_dropout > 0 else None,
            embed_dim=embed_dim,
            image_size=vision_cfg.image_size,
        )
    elif isinstance(vision_cfg.layers, (tuple, list)):
        vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
        visual = ModifiedResNet(
            layers=vision_cfg.layers,
            output_dim=embed_dim,
            heads=vision_heads,
            image_size=vision_cfg.image_size,
            width=vision_cfg.width,
        )
    else:
        vision_heads = vision_cfg.width // vision_cfg.head_width
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
        visual = VisionTransformer(
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            ls_init_value=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            input_patchnorm=vision_cfg.input_patchnorm,
            global_average_pool=vision_cfg.global_average_pool,
            attentional_pool=vision_cfg.attentional_pool,
            n_queries=vision_cfg.n_queries,
            attn_pooler_heads=vision_cfg.attn_pooler_heads,
            output_tokens=vision_cfg.output_tokens,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
    return visual

def _build_vision_tower_Mul(
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None
):
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
    # memory efficient in recent PyTorch releases (>= 1.10).
    # NOTE: timm models always use native GELU regardless of quick_gelu flag.
    act_layer = QuickGELU if quick_gelu else nn.GELU

    if vision_cfg.timm_model_name:
        visual = TimmModel(
            vision_cfg.timm_model_name,
            pretrained=vision_cfg.timm_model_pretrained,
            pool=vision_cfg.timm_pool,
            proj=vision_cfg.timm_proj,
            proj_bias=vision_cfg.timm_proj_bias,
            drop=vision_cfg.timm_drop,
            drop_path=vision_cfg.timm_drop_path,
            patch_drop=vision_cfg.patch_dropout if vision_cfg.patch_dropout > 0 else None,
            embed_dim=embed_dim,
            image_size=vision_cfg.image_size,
        )
    elif isinstance(vision_cfg.layers, (tuple, list)):
        vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
        visual = ModifiedResNet(
            layers=vision_cfg.layers,
            output_dim=embed_dim,
            heads=vision_heads,
            image_size=vision_cfg.image_size,
            width=vision_cfg.width,
        )
    else:
        vision_heads = vision_cfg.width // vision_cfg.head_width
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
        visual = VisionTransformer_Mul(
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            ls_init_value=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            input_patchnorm=vision_cfg.input_patchnorm,
            global_average_pool=vision_cfg.global_average_pool,
            attentional_pool=vision_cfg.attentional_pool,
            n_queries=vision_cfg.n_queries,
            attn_pooler_heads=vision_cfg.attn_pooler_heads,
            output_tokens=vision_cfg.output_tokens,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
    return visual

def _build_text_tower(
        embed_dim: int,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
):
    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)

    if text_cfg.hf_model_name:
        text = HFTextEncoder(
            text_cfg.hf_model_name,
            output_dim=embed_dim,
            proj=text_cfg.proj,
            pooler_type=text_cfg.pooler_type,
            pretrained=text_cfg.hf_model_pretrained,
            output_tokens=text_cfg.output_tokens,
        )
    else:
        act_layer = QuickGELU if quick_gelu else nn.GELU
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm

        text = TextTransformer(
            context_length=text_cfg.context_length,
            vocab_size=text_cfg.vocab_size,
            width=text_cfg.width,
            heads=text_cfg.heads,
            layers=text_cfg.layers,
            ls_init_value=text_cfg.ls_init_value,
            output_dim=embed_dim,
            embed_cls=text_cfg.embed_cls,
            output_tokens=text_cfg.output_tokens,
            pad_id=text_cfg.pad_id,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
    return text

class BatchNormPoint(nn.Module):
    def __init__(self, feat_size):
        super().__init__()
        self.feat_size = feat_size
        self.bn = nn.BatchNorm1d(feat_size)

    def forward(self, x):
        assert len(x.shape) == 3
        s1, s2, s3 = x.shape[0], x.shape[1], x.shape[2]
        assert s3 == self.feat_size
        x = x.view(s1 * s2, self.feat_size)
        x = self.bn(x)
        return x.view(s1, s2, s3)

class CLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)

        text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.transformer = text.transformer
        self.context_length = text.context_length
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.register_buffer('attn_mask', text.attn_mask, persistent=False)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.transformer.grad_checkpointing = enable

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return F.normalize(x, dim=-1) if normalize else x

    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
    ):
        image_features = self.encode_image(image, normalize=True) if image is not None else None
        text_features = self.encode_text(text, normalize=True) if text is not None else None
        if self.output_dict:
            return {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp()
            }
        return image_features, text_features, self.logit_scale.exp()

# ... existing code ...

class TransformerBasicHead(nn.Module):
    """
    Basic Transformer Head. No pool.
    
    这个类实现了一个基础的Transformer头部，包含三层全连接层和批归一化层，
    用于将输入特征映射到类别预测结果。
    
    Args:
        dim_in (int): 输入特征的维度
        num_classes (int): 输出类别数量
    """

    def __init__(
        self,
        dim_in,
        num_classes
    ):
        super(TransformerBasicHead, self).__init__()
        # 定义三层全连接层和对应的批归一化层
        self.projection1 = nn.Linear(dim_in, 128, bias=True)
        self.projection2 = nn.Linear(128, 64, bias=True)
        self.projection3 = nn.Linear(64, num_classes, bias=True)
        self.bn1 = nn.BatchNorm1d(dim_in)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)

    def forward(self, x):
        """
        前向传播函数
        
        输入特征经过三层全连接层处理，每层后面都跟有ReLU激活函数和批归一化，
        最后通过sigmoid函数得到概率输出。
        
        Args:
            x (Tensor): 输入张量，形状为(batch_size, dim_in)
            
        Returns:
            Tensor: 输出张量，形状为(batch_size, num_classes)，值域为[0,1]
        """
        # 第一层：线性变换 -> ReLU激活 -> 批归一化
        x = self.projection1(x)
        x = F.relu(x, inplace=True)
        x = self.bn2(x)
        # 第二层：线性变换 -> ReLU激活 -> 批归一化
        x = self.projection2(x)
        x = F.relu(x, inplace=True)
        x = self.bn3(x)
        # 第三层：线性变换 -> sigmoid激活
        x = self.projection3(x)
        return torch.sigmoid(x)

# ... existing code ...
class Adapter(nn.Module):
    """Adapter模块用于特征变换，通过两个全连接层和ReLU激活函数实现。
    
    Args:
        c_in (int): 输入特征维度
        reduction (int, optional): 降维比例因子，默认为4
    """
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """前向传播函数
        
        Args:
            x (Tensor): 输入张量
            
        Returns:
            Tensor: 经过Adapter变换后的输出张量
        """
        x = self.fc(x)
        return x



class InCTRL(nn.Module):
    """
    InCTRL 模型类，用于图像与文本的联合编码与推理。

    参数:
        args: 配置参数对象。
        embed_dim (int): 嵌入维度。
        vision_cfg (CLIPVisionCfg): 视觉塔的配置参数。
        text_cfg (CLIPTextCfg): 文本塔的配置参数。
        quick_gelu (bool): 是否使用 Quick GELU 激活函数，默认为 False。
        cast_dtype (Optional[torch.dtype]): 模型参数的数据类型，默认为 None。
        output_dict (bool): 是否以字典形式输出结果，默认为 False。
    """

    def __init__(
            self,
           
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict
        # 构建视觉编码器
        self.visual = _build_vision_tower_Mul(embed_dim, vision_cfg, quick_gelu, cast_dtype)

        # 构建文本编码器并提取组件
        text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.transformer = text.transformer
        self.context_length = text.context_length
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.register_buffer('attn_mask', text.attn_mask, persistent=False)

        # 添加适配器和差异头模块
        self.adapter = Adapter(768, 4)
        self.diff_head = TransformerBasicHead(225, 1)
        self.diff_head_ref = TransformerBasicHead(768, 1)

        # 冻结视觉和文本编码器的参数
        for p in self.visual.parameters():
            p.requires_grad = False

        for p in text.parameters():
            p.requires_grad = False

    def encode_image(self, image, out_layers: list = [7, 9, 11], normalize: bool = False):
        """
        编码图像特征。

        参数:
            image (torch.Tensor): 输入图像张量。
            out_layers (list): 输出中间层索引列表，默认为 [7, 9, 11]。
            normalize (bool): 是否对输出特征进行归一化，默认为 False。

        返回:
            图像编码后的特征张量。
        """
        features = self.visual.forward(image, out_layers)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        """
        编码文本特征。

        参数:
            text (torch.Tensor): 输入文本张量。
            normalize (bool): 是否对输出特征进行归一化，默认为 False。

        返回:
            文本编码后的特征张量。
        """
        cast_dtype = self.transformer.get_cast_dtype()
        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # 取每个序列中 eot_token 的特征
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return F.normalize(x, dim=-1) if normalize else x

    def forward(self, tokenizer, image: Optional[torch.Tensor] = None, text: Optional[torch.Tensor] = None, normal_list = None):
        """
        前向传播函数，处理图像和文本输入，计算最终得分。

        参数:
            tokenizer: 文本分词器。
            image (Optional[torch.Tensor]): 输入图像张量，默认为 None。
            text (Optional[torch.Tensor]): 输入文本张量，默认为 None。
            normal_list: 正常样本图像列表，默认为 None。

        返回:
            final_score (torch.Tensor): 最终得分。
            img_ref_score (torch.Tensor): 图像参考得分。
        """
        # 处理输入图像和正常图像
        if normal_list == None:
            img = image[0].cuda(non_blocking=True)
            normal_image = image[1:]
            normal_image = torch.stack(normal_image)
            shot, b, _, _, _ = normal_image.shape
            normal_image = normal_image.reshape(-1, 3,224, 224).cuda(non_blocking=True)
        else:
            img = image[0].cuda(non_blocking=True)
            normal_image = normal_list
            normal_image = torch.stack(normal_image)
            normal_image = normal_image.unsqueeze(1)
            b = len(img)
            normal_image = normal_image.repeat(1, b, 1, 1, 1)
            shot, _, _, _, _ = normal_image.shape
            normal_image = normal_image.reshape(-1, 3, 224, 224).cuda(non_blocking=True)
        print(f'shot:{shot}, b:{b}, normal_image.shape:{normal_image.shape}')
        print(f'img.shape:{img.shape}, normal_image.shape:{normal_image.shape}')
        # 编码图像特征
        token, Fp_list, Fp = self.encode_image(img, normalize=False)
        token_n, Fp_list_n, Fp_n = self.encode_image(normal_image, normalize=False)
        # print(f'Fp_list.shape:{Fp_list.shape}, Fp_list_n.shape:{Fp_list_n.shape}')
        # print(f'token.shape:{token.shape}, token_n.shape:{token_n.shape}')
        # print(f'Fp.shape:{Fp.shape}, Fp_n.shape:{Fp_n.shape}')
        # 整理特征维度
        Fp_list = torch.stack(Fp_list)
        Fp_list_n = torch.stack(Fp_list_n)

        Fp_list = Fp_list[:, :, 1:, :]
        Fp_list_n = Fp_list_n[:, :, 1:, :]

        # Fp_list = Fp_list.reshape(b, 3, 225, -1)
        # Fp_list_n = Fp_list_n.reshape(b, 3, 225 * shot, -1)
        Fp_list = Fp_list.reshape(b, 3, 256, -1)
        Fp_list_n = Fp_list_n.reshape(b, 3, 256 * shot, -1)

        token_n = token_n.reshape(b, shot, -1)

        # 2. 计算图像级残差 (Image-level Residual)
        token_ad = self.adapter.forward(token)
        token_n = self.adapter.forward(token_n)
        token_n = torch.mean(token_n, dim=1)
        token_ref = token_n - token_ad

        text_score = []
        max_diff_score = []
        patch_ref_map = []

        # 遍历每个样本进行处理
        for i in range(len(token)):
            # 3. 计算多层 Patch 级残差 (Multi-layer Patch-level Residual)
            Fp = Fp_list[i, :, :, :]
            Fp_n = Fp_list_n[i, :, :, :]

            Fp_map = list()
            # 计算 patch 级别的参考映射
            for j in range(len(Fp)):
                tmp_x = Fp[j, :, :]
                tmp_n = Fp_n[j, :, :]
                am_fp = list()
                for k in range(len(tmp_x)):
                    tmp = tmp_x[k]
                    tmp = tmp.unsqueeze(0)
                    tmp_n = tmp_n / tmp_n.norm(dim=-1, keepdim=True)
                    tmp = tmp / tmp.norm(dim=-1, keepdim=True)
                    s = (0.5 * (1 - (tmp @ tmp_n.T))).min(dim=1).values
                    am_fp.append(s)
                am_fp = torch.stack(am_fp)
                Fp_map.append(am_fp)
            Fp_map = torch.stack(Fp_map)
            Fp_map = torch.mean(Fp_map.squeeze(2), dim=0)
            patch_ref_map.append(Fp_map)
            score = Fp_map.max(dim=0).values
            max_diff_score.append(score)

             # 4. 结合文本先验 (Text Prior)
            image_feature = token[i]
            image_feature = image_feature.unsqueeze(0)
            image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)

            obj_type = text[i]
            reconstructed_texts, original_texts = get_texts(obj_type.replace('_', " "))
            pos_features = tokenizer(reconstructed_texts).cuda()
            neg_features = tokenizer(original_texts).cuda()
            pos_features = self.encode_text(pos_features)
            neg_features = self.encode_text(neg_features)
            pos_features = pos_features / pos_features.norm(dim=-1, keepdim=True)
            neg_features = neg_features / neg_features.norm(dim=-1, keepdim=True)
            pos_features = torch.mean(pos_features, dim=0, keepdim=True)
            neg_features = torch.mean(neg_features, dim=0, keepdim=True)
            pos_features = pos_features / pos_features.norm(dim=-1, keepdim=True)
            neg_features = neg_features / neg_features.norm(dim=-1, keepdim=True)
            text_features = torch.cat([pos_features, neg_features], dim=0)
            score = (100 * image_feature @ text_features.T).softmax(dim=-1)
            tmp = score[0, 1]
            text_score.append(tmp)

        # 综合得分计算
        # text_score = torch.stack(text_score).unsqueeze(1)
        # img_ref_score = self.diff_head_ref.forward(token_ref)
        # patch_ref_map = torch.stack(patch_ref_map)
        # holistic_map = text_score + img_ref_score + patch_ref_map
        # hl_score = self.diff_head.forward(holistic_map)

        # hl_score = hl_score.squeeze(1)
        # fg_score = torch.stack(max_diff_score)
        # final_score = (hl_score + fg_score) / 2

        # img_ref_score = img_ref_score.squeeze(1)

        # return final_score, img_ref_score
        # 综合得分计算 (原始代码)
        text_score = torch.stack(text_score).unsqueeze(1)
        img_ref_score = self.diff_head_ref.forward(token_ref)
        patch_ref_map = torch.stack(patch_ref_map)
        holistic_map = text_score + img_ref_score + patch_ref_map
        hl_score = self.diff_head.forward(holistic_map)

        hl_score = hl_score.squeeze(1)
        fg_score = torch.stack(max_diff_score)
        
        # 最终的异常分数 (0到1之间)，维度是 (batch_size,)
        anomaly_score = (hl_score + fg_score) / 2

        # --- 新增代码开始 ---
        # 计算正常分数
        normal_score = 1.0 - anomaly_score

        # 将正常分数和异常分数堆叠成一个二维张量
        # 使用 unsqueeze(1) 将 (batch_size,) 变为 (batch_size, 1) 以便拼接
        output_scores = torch.stack([normal_score, anomaly_score], dim=1)
        # --- 新增代码结束 ---

        img_ref_score = img_ref_score.squeeze(1)

        # 返回新的二维得分张量
        return output_scores, img_ref_score

class CustomTextCLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        self.text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.context_length = self.text.context_length
        self.vocab_size = self.text.vocab_size
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    def lock_text_tower(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        self.text.lock(unlocked_layers, freeze_layer_norm)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        features = self.text(text)
        return F.normalize(features, dim=-1) if normalize else features

    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
    ):
        image_features = self.encode_image(image, normalize=True) if image is not None else None
        text_features = self.encode_text(text, normalize=True) if text is not None else None
        if self.output_dict:
            return {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp()
            }
        return image_features, text_features, self.logit_scale.exp()


def convert_weights_to_lp(model: nn.Module, dtype=torch.float16):
    """Convert applicable model parameters to low-precision (bf16 or fp16)"""

    def _convert_weights(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.to(dtype)
            if l.bias is not None:
                l.bias.data = l.bias.data.to(dtype)

        if isinstance(l, (nn.MultiheadAttention, Attention)):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.to(dtype)

        if isinstance(l, (CLIP, TextTransformer)):
            # convert text nn.Parameter projections
            attr = getattr(l, "text_projection", None)
            if attr is not None:
                attr.data = attr.data.to(dtype)

        if isinstance(l, VisionTransformer):
            # convert vision nn.Parameter projections
            attr = getattr(l, "proj", None)
            if attr is not None:
                attr.data = attr.data.to(dtype)

    model.apply(_convert_weights)


convert_weights_to_fp16 = convert_weights_to_lp  # backwards compat


# used to maintain checkpoint compatibility
def convert_to_custom_text_state_dict(state_dict: dict):
    if 'text_projection' in state_dict:
        # old format state_dict, move text tower -> .text
        new_state_dict = {}
        for k, v in state_dict.items():
            if any(k.startswith(p) for p in (
                'text_projection',
                'positional_embedding',
                'token_embedding',
                'transformer',
                'ln_final',
            )):
                k = 'text.' + k
            new_state_dict[k] = v
        return new_state_dict
    return state_dict


def build_model_from_openai_state_dict(
        state_dict: dict,
        quick_gelu=True,
        cast_dtype=torch.float16,
):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_size = vision_patch_size * grid_size
    else:
        counts: list = [
            len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_size = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    vision_cfg = CLIPVisionCfg(
        layers=vision_layers,
        width=vision_width,
        patch_size=vision_patch_size,
        image_size=image_size,
    )
    text_cfg = CLIPTextCfg(
        context_length=context_length,
        vocab_size=vocab_size,
        width=transformer_width,
        heads=transformer_heads,
        layers=transformer_layers,
    )
    model = CLIP(
        embed_dim,
        vision_cfg=vision_cfg,
        text_cfg=text_cfg,
        quick_gelu=quick_gelu,  # OpenAI models were trained with QuickGELU
        cast_dtype=cast_dtype,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        state_dict.pop(key, None)

    convert_weights_to_fp16(model)  # OpenAI state dicts are partially converted to float16
    model.load_state_dict(state_dict)
    return model.eval()


def trace_model(model, batch_size=256, device=torch.device('cpu')):
    model.eval()
    image_size = model.visual.image_size
    example_images = torch.ones((batch_size, 3, image_size, image_size), device=device)
    example_text = torch.zeros((batch_size, model.context_length), dtype=torch.int, device=device)
    model = torch.jit.trace_module(
        model,
        inputs=dict(
            forward=(example_images, example_text),
            encode_text=(example_text,),
            encode_image=(example_images,)
        ))
    model.visual.image_size = image_size
    return model


def resize_pos_embed(state_dict, model, interpolation: str = 'bicubic', antialias: bool = True):
    # Rescale the grid of position embeddings when loading from state_dict
    old_pos_embed = state_dict.get('visual.positional_embedding', None)
    if old_pos_embed is None or not hasattr(model.visual, 'grid_size'):
        return
    grid_size = to_2tuple(model.visual.grid_size)
    extra_tokens = 1  # FIXME detect different token configs (ie no class token, or more)
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
    if new_seq_len == old_pos_embed.shape[0]:
        return

    if extra_tokens:
        pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[extra_tokens:]
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed
    old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))

    logging.info('Resizing position embedding grid-size from %s to %s', old_grid_size, grid_size)
    pos_emb_img = pos_emb_img.reshape(1, old_grid_size[0], old_grid_size[1], -1).permute(0, 3, 1, 2)
    pos_emb_img = F.interpolate(
        pos_emb_img,
        size=grid_size,
        mode=interpolation,
        antialias=antialias,
        align_corners=False,
    )
    pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], -1)[0]
    if pos_emb_tok is not None:
        new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
    else:
        new_pos_embed = pos_emb_img
    state_dict['visual.positional_embedding'] = new_pos_embed

