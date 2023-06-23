# Copyright (c) 2021 Shuai Wang (wsstriving@gmail.com)
#               2022 Zhengyang Chen (chenzhengyang117@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''ResNet in PyTorch.

Some modifications from the original architecture:
1. Smaller kernel size for the input layer
2. Smaller number of Channels
3. No max_pooling involved

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import wespeaker.models.pooling_layers as pooling_layers




class SBSA(nn.Module):
    
    def __init__(self, in_planes, bands=1):
        super(SBSA, self).__init__()
        self.ln_q = nn.Conv2d(in_planes,
                               in_planes,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.ln_k = nn.Conv2d(in_planes,
                               in_planes,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.ln_v = nn.Conv2d(in_planes,
                               in_planes,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        
    def forward(self, x):
        [n_batch,n_planes,n_freq,n_frame] = x.size()
        
        q = self.ln_q(x).permute(0,3,2,1).contiguous().view(n_batch,n_frame,n_freq*n_planes)
        k = self.ln_k(x).permute(0,3,2,1).contiguous().view(n_batch,n_frame,n_freq*n_planes)
        v = self.ln_v(x).permute(0,3,2,1).contiguous().view(n_batch,n_frame,n_freq*n_planes)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(n_freq*n_planes)
        attn = torch.softmax(scores, dim=-1)
        v = torch.matmul(attn,v).view(n_batch,n_frame,n_freq,n_planes).permute(0,3,2,1).contiguous()
        
        return v

class GDCNN(nn.Module):
    
    def __init__(self, in_planes, planes, ratio=4, head=2, bands=1, local_size=7, stride=1):
        super(GDCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = nn.Conv2d(in_planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.conv_reduc = nn.Conv2d(in_planes,
                               in_planes//ratio,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn_reduc = nn.BatchNorm2d(in_planes//ratio)
        
        self.sbsa1 = SBSA(in_planes//ratio,bands)
        
        self.sbsa2 = SBSA(in_planes//ratio,bands) #two attention heads
        
        self.conv_sbsa_exc = nn.Conv2d(in_planes//ratio*head,
                               in_planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn_sbsa = nn.BatchNorm2d(in_planes)
        
        self.mws = nn.Conv2d(in_planes//ratio,
                               in_planes//ratio,
                               kernel_size=local_size,
                               stride=1,
                               padding=(local_size-1)//2,
                               bias=False,
                               groups=in_planes//ratio,
                               )
        self.mws_weights = torch.Tensor(np.zeros([in_planes//ratio,1,local_size,local_size])+1/(local_size**2))
        self.mws.weight = torch.nn.Parameter(self.mws_weights)
                               
        self.conv_mws_exc = nn.Conv2d(in_planes//ratio,
                               in_planes,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn_mws = nn.BatchNorm2d(in_planes)
        
        self.conv_sks = nn.Conv2d(in_planes,
                               planes,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn_sks = nn.BatchNorm2d(planes)
        
        
        
    def forward(self, x):

        acve_r = F.relu(self.bn_reduc(self.conv_reduc(x)))
        
        sbsa = torch.cat((self.sbsa1(acve_r),self.sbsa2(acve_r)),1)
        sbsa = F.relu(self.bn_sbsa(self.conv_sbsa_exc(sbsa)))
        
        mws = self.mws(acve_r)
        mws = F.relu(self.bn_mws(self.conv_mws_exc(mws)))
        
        acve = x + sbsa + mws
        sks = 1.0 + torch.tanh(self.bn_sks(self.conv_sks(acve)))
        
        '''
        equivalence transformation:
        y = K(x)*x, K(x) = K*f(x) => y = K*f(x)*x => y = K*g(x), g(x) = f(x)*x
        where K is the kernel, and K(x) can be viewed as dynamic kernel.
        That is, scaling the kernel values is equivalent to applying scaling factors to feature patches
        '''
        # (abandoned due to computation resource waste)
        # [n_batch,n_planes,n_freq,n_frame] = x.size()
        # x1 = F.unfold(x, kernel_size=3)
        # s1 = F.unfold(sks, kernel_size=3)
        # out1 = self.bn1(self.conv1(F.fold(s1*x1,output_size=(n_freq,n_frame),kernel_size=3)))
        
        # x2 = F.unfold(x, kernel_size=3)
        # s2 = F.unfold(2.0-sks, kernel_size=3)
        # out2 = self.bn1(self.conv2(F.fold(s2*x2,output_size=(n_freq,n_frame),kernel_size=3)))
        
        out1 = self.bn1(self.conv1(sks*x))
        out2 = self.bn2(self.conv2((2.0-sks)*x)) 
       
        return out1+out2
        
class BasicBlock2(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_dynamic=False):
        super(BasicBlock2, self).__init__()
        self.is_dynamic = is_dynamic
        self.conv1 = nn.Conv2d(in_planes,
                               planes,
                               kernel_size=3,
                               stride=(stride,1),
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        if is_dynamic:
            self.conv2 = GDCNN(planes,planes)
            # self.conv2 = DCNN(planes,planes)
        else:
            self.conv2 = nn.Sequential(nn.Conv2d(planes,
                                      planes,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1,
                                      bias=False),
                                      nn.BatchNorm2d(planes))
                               
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=(stride,1),
                          bias=False),
                nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes,
                               self.expansion * planes,
                               kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet2(nn.Module):
    def __init__(self,
                 block,
                 num_blocks,
                 m_channels=32,
                 feat_dim=40,
                 embed_dim=128,
                 pooling_func='TSTP',
                 two_emb_layer=True,
                 is_dynamic=False):
        super(ResNet2, self).__init__()
        self.in_planes = m_channels
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.stats_dim = int(feat_dim / 16) * m_channels * 8
        self.two_emb_layer = two_emb_layer

        self.conv1 = nn.Conv2d(1,
                               m_channels,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(block,
                                       m_channels,
                                       num_blocks[0],
                                       stride=1,
                                       is_dynamic=is_dynamic)
        self.layer2 = self._make_layer(block,
                                       m_channels * 2,
                                       num_blocks[1],
                                       stride=2,
                                       is_dynamic=is_dynamic)
        self.layer3 = self._make_layer(block,
                                       m_channels * 4,
                                       num_blocks[2],
                                       stride=2,
                                       is_dynamic=is_dynamic)
        self.layer4 = self._make_layer(block,
                                       m_channels * 8,
                                       num_blocks[3],
                                       stride=2,
                                       is_dynamic=False)

        self.pool = getattr(pooling_layers, pooling_func)(
            in_dim=self.stats_dim * block.expansion)
        self.pool_out_dim = self.pool.get_out_dim()
        self.seg_1 = nn.Linear(self.pool_out_dim,
                               embed_dim)
        if self.two_emb_layer:
            self.seg_bn_1 = nn.BatchNorm1d(embed_dim, affine=False)
            self.seg_2 = nn.Linear(embed_dim, embed_dim)
        else:
            self.seg_bn_1 = nn.Identity()
            self.seg_2 = nn.Identity()

    def _make_layer(self, block, planes, num_blocks, stride, is_dynamic=True):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, is_dynamic))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)

        x = x.unsqueeze_(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        stats = self.pool(out)

        embed_a = self.seg_1(stats)
        if self.two_emb_layer:
            out = F.relu(embed_a)
            out = self.seg_bn_1(out)
            embed_b = self.seg_2(out)
            return embed_a, embed_b
        else:
            return torch.tensor(0.0), embed_a

class ResNet(nn.Module):
    def __init__(self,
                 block,
                 num_blocks,
                 m_channels=32,
                 feat_dim=40,
                 embed_dim=128,
                 pooling_func='TSTP',
                 two_emb_layer=True,
                 is_dynamic=False):
        super(ResNet, self).__init__()
        self.in_planes = m_channels
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.stats_dim = int(feat_dim / 8) * m_channels * 8
        self.two_emb_layer = two_emb_layer

        self.conv1 = nn.Conv2d(1,
                               m_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(block,
                                       m_channels,
                                       num_blocks[0],
                                       stride=1,
                                       is_dynamic=True)
        self.layer2 = self._make_layer(block,
                                       m_channels * 2,
                                       num_blocks[1],
                                       stride=2,
                                       is_dynamic=True)
        self.layer3 = self._make_layer(block,
                                       m_channels * 4,
                                       num_blocks[2],
                                       stride=2,
                                       is_dynamic=True)
        self.layer4 = self._make_layer(block,
                                       m_channels * 8,
                                       num_blocks[3],
                                       stride=2,
                                       is_dynamic=False)

        self.pool = getattr(pooling_layers, pooling_func)(
            in_dim=self.stats_dim * block.expansion)
        self.pool_out_dim = self.pool.get_out_dim()
        self.seg_1 = nn.Linear(self.pool_out_dim,
                               embed_dim)
        if self.two_emb_layer:
            self.seg_bn_1 = nn.BatchNorm1d(embed_dim, affine=False)
            self.seg_2 = nn.Linear(embed_dim, embed_dim)
        else:
            self.seg_bn_1 = nn.Identity()
            self.seg_2 = nn.Identity()

    def _make_layer(self, block, planes, num_blocks, stride, is_dynamic=False):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, is_dynamic))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)

        x = x.unsqueeze_(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        stats = self.pool(out)

        embed_a = self.seg_1(stats)
        if self.two_emb_layer:
            out = F.relu(embed_a)
            out = self.seg_bn_1(out)
            embed_b = self.seg_2(out)
            return embed_a, embed_b
        else:
            return torch.tensor(0.0), embed_a



def ResNet18(feat_dim, embed_dim, pooling_func='TSTP', two_emb_layer=True):
    return ResNet(BasicBlock, [2, 2, 2, 2],
                  feat_dim=feat_dim,
                  embed_dim=embed_dim,
                  pooling_func=pooling_func,
                  two_emb_layer=two_emb_layer)

def ResNet34_gdcnn(feat_dim, embed_dim, pooling_func='TSTP', two_emb_layer=True):
    return ResNet2(BasicBlock2, [3, 4, 6, 3],
                  feat_dim=feat_dim,
                  embed_dim=embed_dim,
                  pooling_func=pooling_func,
                  two_emb_layer=two_emb_layer,
                  is_dynamic=True)
                  
                  
def ResNet34(feat_dim, embed_dim, pooling_func='TSTP', two_emb_layer=True):
    return ResNet(BasicBlock, [3, 4, 6, 3],
                  feat_dim=feat_dim,
                  embed_dim=embed_dim,
                  pooling_func=pooling_func,
                  two_emb_layer=two_emb_layer)

def ResNet50(feat_dim, embed_dim, pooling_func='TSTP', two_emb_layer=True):
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  feat_dim=feat_dim,
                  embed_dim=embed_dim,
                  pooling_func=pooling_func,
                  two_emb_layer=two_emb_layer)


def ResNet101(feat_dim, embed_dim, pooling_func='TSTP', two_emb_layer=True):
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  feat_dim=feat_dim,
                  embed_dim=embed_dim,
                  pooling_func=pooling_func,
                  two_emb_layer=two_emb_layer)


def ResNet152(feat_dim, embed_dim, pooling_func='TSTP', two_emb_layer=True):
    return ResNet(Bottleneck, [3, 8, 36, 3],
                  feat_dim=feat_dim,
                  embed_dim=embed_dim,
                  pooling_func=pooling_func,
                  two_emb_layer=two_emb_layer)


def ResNet221(feat_dim, embed_dim, pooling_func='TSTP', two_emb_layer=True):
    return ResNet(Bottleneck, [6, 16, 48, 3],
                  feat_dim=feat_dim,
                  embed_dim=embed_dim,
                  pooling_func=pooling_func,
                  two_emb_layer=two_emb_layer)


def ResNet293(feat_dim, embed_dim, pooling_func='TSTP', two_emb_layer=True):
    return ResNet(Bottleneck, [10, 20, 64, 3],
                  feat_dim=feat_dim,
                  embed_dim=embed_dim,
                  pooling_func=pooling_func,
                  two_emb_layer=two_emb_layer)


if __name__ == '__main__':
    model = ResNet34_gdcnn(feat_dim=80, embed_dim=256, pooling_func='TSTP')
    model.eval()
    y = model(torch.randn(10, 200, 80))
    print(y[-1].size())

    num_params = sum(p.numel() for p in model.parameters())
    print("{} M".format(num_params / 1e6))

    # from thop import profile
    # x_np = torch.randn(1, 200, 80)
    # flops, params = profile(model, inputs=(x_np, ))
    # print("FLOPS: {} G, Params: {} M".format(flops / 1e9, params / 1e6))
