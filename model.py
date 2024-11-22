#%%
import torch
import torch.nn as nn
from functools import partial
from timm.models import register_model
from braincog.base.node.node import *
from braincog.base.encoder.encoder import *
from braincog.model_zoo.base_module import BaseModule, BaseConvModule

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
            self.node = partial(self.node, layer_by_layer=True, **kwargs, step=step)

        self.left = nn.Sequential(
                    BaseConvModule(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
                    nn.BatchNorm2d(outchannel),
                    nn.ReLU(inplace=True),
                    BaseConvModule(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(outchannel)
                )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        self.reset()
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

@register_model
class ResNet_18(BaseModule):
    def __init__(self, step, num_classes=10, in_channels=3, *args, **kargs):
        self.encode_type = 'direct'
        self.step = step
        super(ResNet_18, self).__init__(step=step, encode_type=self.encode_type, layer_by_layer=True, temporal_flatten=False)
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, self.inchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.inchannel),
            nn.ReLU(),
        )
        block = BasicBlock
        self.layer1 = self.make_layer(block, 64, 2, stride=1)
        self.layer2 = self.make_layer(block, 128, 2, stride=2)
        self.layer3 = self.make_layer(block, 256, 2, stride=2)
        self.layer4 = self.make_layer(block, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride, step=self.step, encode_type=self.encode_type))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = rearrange(out, '(t b) c -> t b c', t=self.step).mean(0)
        return out
    
class TwoStreamSNN(nn.Module):
    def __init__(self, flow_channels, rgb_channels, num_classes, step):
        super().__init__()
        self.rgb_model = ResNet_18(step, 512, rgb_channels)
        self.flow_model = ResNet_18(step, 512, flow_channels)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
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
    pass
    #%%
    data = torch.rand(32, 3, 32, 32)
    model = BasicBlock(3, 8)
    print(model(data).shape)

    #%%
    data = torch.rand(32, 8, 3, 32, 32)
    lbl = True
    model = ResNet_18(8, 101, 3)
    out = model(data)
    print(out.shape)

    #%%
    # test two stream, 101 classes
    flow_data = torch.rand(2, 10, 8, 32, 32)
    rgb_data = torch.rand(2, 10, 3, 32, 32)
    labels = torch.randint(0, 101, (2,))
    model = TwoStreamSNN(8, 3, 101, 10)
    out = model(rgb_data, flow_data)
    print(out.shape)
    
    #%%
    loss = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    for i in range(100):
        opt.zero_grad()
        out = model(rgb_data, flow_data)
        loss_val = loss(out, labels)
        loss_val.backward()
        opt.step()
        print(loss_val.item())
        break