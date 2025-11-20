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
"""blocks net."""
import torch.nn.functional as F
import torch.nn as nn
import torch


class FeatureFusionBlock(nn.Module):
    """FeatureFusionBlock."""
    def __init__(self, features):
        super(FeatureFusionBlock, self).__init__()
        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs):
        output = xs[0]
        if len(xs) == 2:
            # output = torch.cat([output,self.resConfUnit1(xs[1])],1)
            # print('output',output.shape)
            # print('self.resConfUnit1(xs[1])',self.resConfUnit1(xs[1]).shape)
            output += self.resConfUnit1(xs[1])
        output = self.resConfUnit2(output)
        # size_x = output.shape[2] * 2
        # size_y = output.shape[3] * 2
        # print(size_x)
        # print(size_y)
        # print(output.shape)
        output = F.interpolate(output,scale_factor=2,mode='bilinear')
        return output
    

class FeatureFusionBlock_BN(nn.Module):
    """FeatureFusionBlock."""
    def __init__(self, features):
        super(FeatureFusionBlock_BN, self).__init__()
        self.resConfUnit1 = ResidualConvUnit1(features)
        self.resConfUnit2 = ResidualConvUnit1(features)

    def forward(self, *xs):
        output = xs[0]
        if len(xs) == 2:
            # output = torch.cat([output,self.resConfUnit1(xs[1])],1)
            # print('output',output.shape)
            # print('self.resConfUnit1(xs[1])',self.resConfUnit1(xs[1]).shape)
            output += self.resConfUnit1(xs[1])
        output = self.resConfUnit2(output)
        # size_x = output.shape[2] * 2
        # size_y = output.shape[3] * 2
        # print(size_x)
        # print(size_y)
        # print(output.shape)
        output = F.interpolate(output,scale_factor=2,mode='bilinear')
        return output
    

class FeatureFusionBlock6(nn.Module):
    """FeatureFusionBlock."""
    def __init__(self, features):
        super(FeatureFusionBlock6, self).__init__()
        self.resConfUnit1 = ResidualConvUnit6(features)
        self.resConfUnit2 = ResidualConvUnit6(features)

    def forward(self, *xs):
        output = xs[0]
        if len(xs) == 2:
            # output = torch.cat([output,self.resConfUnit1(xs[1])],1)
            # print('output',output.shape)
            # print('self.resConfUnit1(xs[1])',self.resConfUnit1(xs[1]).shape)
            output += self.resConfUnit1(xs[1])
        output = self.resConfUnit2(output)
        # size_x = output.shape[2] * 2
        # size_y = output.shape[3] * 2
        # print(size_x)
        # print(size_y)
        # print(output.shape)
        output = F.interpolate(output,scale_factor=2,mode='bilinear')
        return output


class ResidualConvUnit(nn.Module):
    """ResidualConvUnit."""
    def __init__(self, features):
        super().__init__()
        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, bias=True,
            padding=1
        )
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, bias=True,
            padding=1
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        # print('x',x.shape)
        out = self.relu(x)
        out = self.conv1(out)
        # print('out1',out.shape)
        out = self.relu(out)
        out = self.conv2(out)
        # print('out2',out.shape)

        return out + x
    


class ResidualConvUnit1(nn.Module):
    """ResidualConvUnit."""
    def __init__(self, features):
        super().__init__()
        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, bias=True,
            padding=1
        )
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, bias=True,
            padding=1
        )
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(features)

    def forward(self, x):
        # print('x',x.shape)
       
        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)
        # print('out1',out.shape)
       
        out = self.conv2(out)
        out = self.bn(out)
        out = self.relu(out)
        # print('out2',out.shape)

        return out + x
    

class ResidualConvUnit6(nn.Module):
    """ResidualConvUnit."""
    def __init__(self, features):
        super().__init__()
        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, bias=True,
            padding=1
        )
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, bias=True,
            padding=1
        )
        self.relu = nn.ReLU6()

    def forward(self, x):
        # print('x',x.shape)
        out = self.relu(x)
        out = self.conv1(out)
        # print('out1',out.shape)
        out = self.relu(out)
        out = self.conv2(out)
        # print('out2',out.shape)

        return out + x
    


class ResidualConvUnit_mbv2(nn.Module):
    """ResidualConvUnit."""
    def __init__(self, features):
        super().__init__()
        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, bias=True,
            padding=1
        )
        self.conv2 = nn.Conv2d(
            32, 32, kernel_size=3, stride=1, bias=True,
            padding=1
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        print('x',x.shape)
        out = self.relu(x)
        out = self.conv1(out)
        # print('out1',out.shape)
        out = self.relu(out)
        out = self.conv2(out)
        # print('out2',out.shape)

        return out + x
    


class FeatureFusionBlock_small2(nn.Module):
    """FeatureFusionBlock."""
    def __init__(self, features):
        super(FeatureFusionBlock_small2, self).__init__()
        # self.resConfUnit1 = ResidualConvUnit_small2(features)
        # self.resConfUnit2 = ResidualConvUnit_small2(features)

    def forward(self, *xs):
        output = xs[0]
        # if len(xs) == 2:
        #     output += self.resConfUnit1(xs[1])
        # output = self.resConfUnit2(output)
        # size_x = output.shape[2] * 2
        # size_y = output.shape[3] * 2
        # print(size_x)
        # print(size_y)
        # print(output.shape)
        output = F.interpolate(output,scale_factor=2,mode='bilinear')
        return output


class ResidualConvUnit_small2(nn.Module):
    """ResidualConvUnit."""
    def __init__(self, features):
        super().__init__()
        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, bias=True,
            padding=1
        )
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, bias=True,
            padding=1
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        # print('x',x.shape)
        out = self.relu(x)
        out = self.conv1(out)
        # print('out1',out.shape)
        out = self.relu(out)
        out = self.conv2(out)
        # print('out2',out.shape)

        return out + x
