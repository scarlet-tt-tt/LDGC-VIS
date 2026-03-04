# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""net."""
from .blocks_ms import FeatureFusionBlock6
import torch
import torch.nn as nn
from .starnet import starnet_s150
import math

def weights_init(m):
    # Initialize kernel weights with Gaussian distributions
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

class MidasNet_small7(nn.Module):
    """Network for monocular depth estimation.
    """
    def __init__(self,expand=False):
        super(MidasNet_small7, self).__init__()
        self.starnet = starnet_s150(pretrained=True)
        self.stem = self.starnet.stem
        self.stages = self.starnet.stages
        
        # self.out_channels = 512 * block.expansion
        features = 16
        out_shape1 = features
        out_shape2 = features
        out_shape3 = features
        out_shape4 = features
        if expand:
            out_shape1 = features * 0.5
            out_shape2 = features * 2 * 0.5
            out_shape3 = features * 4 * 0.5
            out_shape4 = features * 8 * 0.5
        self.layer1_rn_scratch = nn.Conv2d(
            24, out_shape1, kernel_size=3, stride=1, bias=False,
            padding=1
        )
        self.layer2_rn_scratch = nn.Conv2d(
            48, out_shape2, kernel_size=3, stride=1, bias=False,
            padding=1
        )
        self.layer3_rn_scratch = nn.Conv2d(
            96, out_shape3, kernel_size=3, stride=1, bias=False,
            padding=1
        )
        self.layer4_rn_scratch = nn.Conv2d(
            192, out_shape4, kernel_size=3, stride=1, bias=False,
            padding=1
        )
        self.refinenet4_scratch = FeatureFusionBlock6(features)
        self.refinenet3_scratch = FeatureFusionBlock6(features)
        self.refinenet2_scratch = FeatureFusionBlock6(features)
        self.refinenet1_scratch = FeatureFusionBlock6(features)
        self.output_conv_scratch1 = nn.Conv2d(
                features, 16, kernel_size=3, stride=1, bias=True,
                padding=1
            )
        self.output_conv_scratch2 = nn.Sequential(nn.Conv2d(
                16, 8, kernel_size=3, stride=1, bias=True,
                padding=1
            ),
            nn.ReLU6(),
            
            nn.Conv2d(
                8, 1, kernel_size=1, stride=1, bias=True,
                padding=0
            ),
            nn.ReLU6())
        # self.squeeze = ops.Squeeze(1)
        weights_init(self.layer1_rn_scratch)
        weights_init(self.layer2_rn_scratch)
        weights_init(self.layer3_rn_scratch)
        weights_init(self.layer4_rn_scratch)
        weights_init(self.output_conv_scratch1)
        weights_init(self.output_conv_scratch1)

    def forward(self, x):
        """construct pass.

        Args:
            x (tensor): input data (image)

        Returns:
            tensor: depth
        """
        # print('x',x.shape)
        outs = []
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
            outs.append(x)
        # print('outs[0]',outs[0].shape) # 24,64,80
        # print('outs[1]',outs[1].shape)# 48,32,40
        # print('outs[2]',outs[2].shape)#96,16,20
        # print('outs[3]',outs[3].shape)# 192,8,10
        layer_1_rn = self.layer1_rn_scratch(outs[0])
        layer_2_rn = self.layer2_rn_scratch(outs[1])
        layer_3_rn = self.layer3_rn_scratch(outs[2])
        layer_4_rn = self.layer4_rn_scratch(outs[3])

        path_4 = self.refinenet4_scratch(layer_4_rn)
        path_3 = self.refinenet3_scratch(path_4, layer_3_rn)
        path_2 = self.refinenet2_scratch(path_3, layer_2_rn)
        path_1 = self.refinenet1_scratch(path_2, layer_1_rn)
        out = self.output_conv_scratch1(path_1)

        out = torch.nn.functional.interpolate(out,scale_factor=2)
        out = self.output_conv_scratch2(out)
        result = torch.squeeze(out,1)

        return result
    


class MidasNet_small7_weight(nn.Module):
    """Network for monocular depth estimation.
    """
    def __init__(self,expand=False):
        super(MidasNet_small7_weight, self).__init__()
        self.starnet = starnet_s150(pretrained=True)
        self.stem = self.starnet.stem
        self.stages = self.starnet.stages
        
        # self.out_channels = 512 * block.expansion
        features = 16
        out_shape1 = features
        out_shape2 = features
        out_shape3 = features
        out_shape4 = features
        if expand:
            out_shape1 = features * 0.5
            out_shape2 = features * 2 * 0.5
            out_shape3 = features * 4 * 0.5
            out_shape4 = features * 8 * 0.5
        self.layer1_rn_scratch_weight = nn.Conv2d(
            24, out_shape1, kernel_size=3, stride=1, bias=False,
            padding=1
        )
        self.layer2_rn_scratch_weight = nn.Conv2d(
            48, out_shape2, kernel_size=3, stride=1, bias=False,
            padding=1
        )
        self.layer3_rn_scratch_weight = nn.Conv2d(
            96, out_shape3, kernel_size=3, stride=1, bias=False,
            padding=1
        )
        self.layer4_rn_scratch_weight = nn.Conv2d(
            192, out_shape4, kernel_size=3, stride=1, bias=False,
            padding=1
        )
        self.refinenet4_scratch_weight = FeatureFusionBlock6(features)
        self.refinenet3_scratch_weight = FeatureFusionBlock6(features)
        self.refinenet2_scratch_weight = FeatureFusionBlock6(features)
        self.refinenet1_scratch_weight = FeatureFusionBlock6(features)
        self.output_conv_scratch1_weight = nn.Conv2d(
                features, 16, kernel_size=3, stride=1, bias=True,
                padding=1
            )
        self.output_conv_scratch2_weight = nn.Sequential(nn.Conv2d(
                16, 8, kernel_size=3, stride=1, bias=True,
                padding=1
            ),
            nn.ReLU6(),
            
            nn.Conv2d(
                8, 1, kernel_size=1, stride=1, bias=True,
                padding=0
            ),
            nn.ReLU6())
        # self.squeeze = ops.Squeeze(1)
        weights_init(self.layer1_rn_scratch_weight)
        weights_init(self.layer2_rn_scratch_weight)
        weights_init(self.layer3_rn_scratch_weight)
        weights_init(self.layer4_rn_scratch_weight)
        weights_init(self.output_conv_scratch1_weight)
        weights_init(self.output_conv_scratch1_weight)


    def forward(self, x):
        """construct pass.

        Args:
            x (tensor): input data (image)

        Returns:
            tensor: depth
        """
        # print('x',x.shape)
        outs = []
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
            outs.append(x)
        # print('outs[0]',outs[0].shape) # 24,64,80
        # print('outs[1]',outs[1].shape)# 48,32,40
        # print('outs[2]',outs[2].shape)#96,16,20
        # print('outs[3]',outs[3].shape)# 192,8,10
        layer_1_rn_weight = self.layer1_rn_scratch_weight(outs[0])
        layer_2_rn_weight = self.layer2_rn_scratch_weight(outs[1])
        layer_3_rn_weight = self.layer3_rn_scratch_weight(outs[2])
        layer_4_rn_weight = self.layer4_rn_scratch_weight(outs[3])

        path_4_weight = self.refinenet4_scratch_weight(layer_4_rn_weight)
        path_3_weight = self.refinenet3_scratch_weight(path_4_weight, layer_3_rn_weight)
        path_2_weight = self.refinenet2_scratch_weight(path_3_weight, layer_2_rn_weight)

        path_1_weight = self.refinenet1_scratch_weight(path_2_weight, layer_1_rn_weight)
        out_weight = self.output_conv_scratch1_weight(path_1_weight)

        out_weight = torch.nn.functional.interpolate(out_weight,scale_factor=2)
        out_weight = self.output_conv_scratch2_weight(out_weight)
        result_weight = torch.squeeze(out_weight,1)


        return result_weight