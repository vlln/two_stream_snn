#%%
import torch
import torch.nn as nn
from functools import partial
from timm.models import register_model
from braincog.base.node.node import *
from braincog.base.encoder.encoder import *
from braincog.model_zoo.base_module import BaseModule, BaseConvModule
from braincog.datasets import is_dvs_data

#%%
@register_model
class BasicBlock(BaseModule):
    def __init__(self,
                inchannel,
                outchannel,
                stride=1,
                step=8,
                node_type=LIFNode,
                encode_type='direct',
                *args,
                **kwargs):
        super().__init__(step, encode_type, *args, **kwargs)

        self.node = node_type
        if issubclass(self.node, BaseNode):
            self.node = partial(self.node, **kwargs, step=step)

        self.dataset = kwargs['dataset'] if 'dataset' in kwargs else 'dvsc10'
        if not is_dvs_data(self.dataset):
            init_channel = 3
        else:
            init_channel = 2
        
        self.left = nn.Sequential(
                    BaseConvModule(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
                    nn.BatchNorm2d(outchannel),
                    nn.ReLU(inplace=True), #inplace=True表示进行原地操作，一般默认为False，表示新建一个变量存储操作
                    BaseConvModule(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(outchannel)
                )
        self.shortcut = nn.Sequential()
        #论文中模型架构的虚线部分，需要下采样
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        inputs = self.encoder(x)
        self.reset()
        if self.layer_by_layer:
            out = self.left(x)
            out += self.shortcut(x)
            out = F.relu(out)
            return out
        else:
            outputs = []
            for t in range(self.step):
                x = inputs[t]
                out = self.left(x)
                out += self.shortcut(x)
                out = F.relu(out)
                outputs.append(out)
            return sum(outputs) / len(outputs)

#%%
class ResNet_18(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10, in_channels=3):
        super(ResNet_18, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):  # 3*32*32
        out = self.conv1(x)  # 64*32*32
        out = self.layer1(out)  # 64*32*32
        out = self.layer2(out)  # 128*16*16
        out = self.layer3(out)  # 256*8*8
        out = self.layer4(out)  # 512*4*4
        out = F.avg_pool2d(out, 4)  # 512*1*1
        out = out.view(out.size(0), -1)  # 512
        out = self.fc(out)
        return out
    
#%%
class TwoStreamSNN(nn.Module):
    def __init__(self, flow_channels, rgb_channels, num_classes, step, layer_by_layer, node_type, datasets, **kwargs):
        super().__init__()
        block = partial(BasicBlock, step=step, layer_by_layer=layer_by_layer, node_type=node_type, datasets=datasets)
        self.rgb_model = ResNet_18(block, 2048, 3)
        self.flow_model = ResNet_18(block, 2048, 8)
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, rgb, flow):
        rgb_out = self.rgb_model(rgb)
        flow_out = self.flow_model(flow)
        rgb_out = F.softmax(rgb_out, dim=1)
        flow_out = F.softmax(flow_out, dim=1)
        out = (rgb_out + flow_out) / 2
        out = self.classifier(out)
        return out

if __name__ == '__main__':
    # test backbone
    data = torch.rand(2, 3, 32, 32)
    lbl = True
    block = partial(BasicBlock, step=8, layer_by_layer=lbl, node_type=LIFNode, datasets='dvsc10')
    model = ResNet_18(block, 10)
    out = model(data)

    # test two stream
    flow_data = torch.rand(2, 8, 32, 32)
    rgb_data = torch.rand(2, 3, 32, 32)
    model = TwoStreamSNN(8, 3, 10, 8, True, LIFNode, 'dvsc10')
    out = model(rgb_data, flow_data)