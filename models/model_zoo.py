import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class _Swish(torch.autograd.Function):
    """Custom implementation of swish."""

    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish(nn.Module):
    """Module using custom implementation."""

    def forward(self, input_tensor):
        return _Swish.apply(input_tensor)


custom_softplus = nn.Softplus(beta=10)


def softplus(data):
    return custom_softplus(data)


class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        # 记录该模型的名字
        self.model_name = str(type(self))

    def load(self, path, map_location=None):
        self.load_state_dict(torch.load(path, map_location))

    def save(self, name=None):
        if name is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def no_grad(self):
        for param in self.parameters():
            param.requires_grad = False

    def with_grad(self):
        for param in self.parameters():
            param.requires_grad = True

    def clear_grad(self):
        for param in self.parameters():
            param.grad = None


# CNN model used in Madry's PGD paper
class MadryCNN(BasicModule):
    def __init__(self, activation_fn: nn.Module = nn.ReLU):
        super(MadryCNN, self).__init__()
        self.model_name = 'TwoLayerModel'
        self.conv1 = nn.Sequential(
            nn.Conv2d(  # (1,28,28)
                in_channels=1,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),  # (32,28,28)
            activation_fn(beta=10) if activation_fn == nn.Softplus else activation_fn(),
            nn.MaxPool2d(kernel_size=2),  # (32,14,14)
            # nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            # activation_fn(beta=10) if activation_fn == nn.Softplus else activation_fn(),  # (64,14,14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(  # (32,14,14)
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2
            ),  # (64,14,14)
            activation_fn(beta=10) if activation_fn == nn.Softplus else activation_fn(),  # (64,14,14)
            nn.MaxPool2d(kernel_size=2),  # (64,7,7)
            # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            # activation_fn(beta=10) if activation_fn == nn.Softplus else activation_fn(),  # (64,14,14)
        )
        self.linear1 = nn.Sequential(
            nn.Linear(64 * 7 * 7, 1024),
            activation_fn(beta=10) if activation_fn == nn.Softplus else activation_fn(),
        )
        self.out = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        output = self.out(x)
        return output


# CNN Model used in TRADES paper
class TradesCNN(BasicModule):
    def __init__(self, drop=0.5):
        super(TradesCNN, self).__init__()

        self.num_channels = 1
        self.num_labels = 10

        activ = nn.ReLU(True)

        self.feature_extractor = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(self.num_channels, 32, 3)),
            ('relu1', activ),
            ('conv2', nn.Conv2d(32, 32, 3)),
            ('relu2', activ),
            ('maxpool1', nn.MaxPool2d(2, 2)),
            ('conv3', nn.Conv2d(32, 64, 3)),
            ('relu3', activ),
            ('conv4', nn.Conv2d(64, 64, 3)),
            ('relu4', activ),
            ('maxpool2', nn.MaxPool2d(2, 2)),
        ]))

        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(64 * 4 * 4, 200)),
            ('relu1', activ),
            ('drop', nn.Dropout(drop)),
            ('fc2', nn.Linear(200, 200)),
            ('relu2', activ),
            ('fc3', nn.Linear(200, self.num_labels)),
        ]))

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.classifier.fc3.weight, 0)
        nn.init.constant_(self.classifier.fc3.bias, 0)

    def forward(self, input):
        features = self.feature_extractor(input)
        logits = self.classifier(features.view(-1, 64 * 4 * 4))
        return logits


# Modules for Wide-ResNet
class BasicUnit(nn.Module):
    def __init__(self, channels: int, dropout: float, activation_fn: nn.Module):
        super(BasicUnit, self).__init__()
        self.block = nn.Sequential(OrderedDict([
            ("0_normalization", nn.BatchNorm2d(channels)),
            ("1_activation", activation_fn(beta=10) if activation_fn == nn.Softplus else activation_fn()),
            ("2_convolution", nn.Conv2d(channels, channels, (3, 3), stride=1, padding=1, bias=False)),
            ("3_normalization", nn.BatchNorm2d(channels)),
            ("4_activation", activation_fn(beta=10) if activation_fn == nn.Softplus else activation_fn()),
            ("5_dropout", nn.Dropout(dropout, inplace=True)),
            ("6_convolution", nn.Conv2d(channels, channels, (3, 3), stride=1, padding=1, bias=False)),
        ]))

    def forward(self, x):
        return x + self.block(x)


class DownsampleUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, dropout: float, activation_fn: nn.Module):
        super(DownsampleUnit, self).__init__()
        self.norm_act = nn.Sequential(OrderedDict([
            ("0_normalization", nn.BatchNorm2d(in_channels)),
            ("1_activation", activation_fn(beta=10) if activation_fn == nn.Softplus else activation_fn()),
        ]))
        self.block = nn.Sequential(OrderedDict([
            ("0_convolution", nn.Conv2d(in_channels, out_channels, (3, 3), stride=stride, padding=1, bias=False)),
            ("1_normalization", nn.BatchNorm2d(out_channels)),
            ("2_activation", activation_fn(beta=10) if activation_fn == nn.Softplus else activation_fn()),
            ("3_dropout", nn.Dropout(dropout, inplace=True)),
            ("4_convolution", nn.Conv2d(out_channels, out_channels, (3, 3), stride=1, padding=1, bias=False)),
        ]))
        self.downsample = nn.Conv2d(in_channels, out_channels, (1, 1), stride=stride, padding=0, bias=False)

    def forward(self, x):
        x = self.norm_act(x)
        return self.block(x) + self.downsample(x)


class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, depth: int, dropout: float,
                 activation_fn: nn.Module):
        super(Block, self).__init__()
        self.block = nn.Sequential(
            DownsampleUnit(in_channels, out_channels, stride, dropout, activation_fn=activation_fn),
            *(BasicUnit(out_channels, dropout, activation_fn=activation_fn) for _ in range(depth))
        )

    def forward(self, x):
        return self.block(x)


class WideResNet(BasicModule):
    def __init__(self, depth: int, width_factor: int, dropout: float, in_channels: int, labels: int,
                 activation_fn: nn.Module = nn.ReLU):
        super(WideResNet, self).__init__()

        self.filters = [16, 1 * 16 * width_factor, 2 * 16 * width_factor, 4 * 16 * width_factor]
        self.block_depth = (depth - 4) // (3 * 2)

        self.f = nn.Sequential(OrderedDict([
            ("0_convolution", nn.Conv2d(in_channels, self.filters[0], (3, 3), stride=1, padding=1, bias=False)),
            ("1_block", Block(self.filters[0], self.filters[1], 1, self.block_depth, dropout, activation_fn)),
            ("2_block", Block(self.filters[1], self.filters[2], 2, self.block_depth, dropout, activation_fn)),
            ("3_block", Block(self.filters[2], self.filters[3], 2, self.block_depth, dropout, activation_fn)),
            ("4_normalization", nn.BatchNorm2d(self.filters[3])),
            ("5_activation", activation_fn(beta=10) if activation_fn == nn.Softplus else activation_fn()),
            ("6_pooling", nn.AvgPool2d(kernel_size=8)),
            ("7_flattening", nn.Flatten()),
            ("8_classification", nn.Linear(in_features=self.filters[3], out_features=labels)),
        ]))

        self._initialize()

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.zero_()
                m.bias.data.zero_()

    def forward(self, x):
        return self.f(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, activation_fn=nn.ReLU()):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.activation_fn = activation_fn
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.activation_fn(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activation_fn(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, activation_fn=nn.ReLU()):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.activation_fn = activation_fn
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.activation_fn(self.bn1(self.conv1(x)))
        out = self.activation_fn(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.activation_fn(out)
        return out


class ResNet(BasicModule):
    def __init__(self, block, num_blocks, num_classes=10, activation_fn=softplus):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, activation_fn=activation_fn)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, activation_fn=activation_fn)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, activation_fn=activation_fn)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, activation_fn=activation_fn)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.activation_fn = activation_fn

    def _make_layer(self, block, planes, num_blocks, stride, activation_fn):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, activation_fn))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.activation_fn(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# Copied from Fast Adversarial Training
# https://github.com/locuslab/fast_adversarial/blob/54f728755e71857b632882ba6d6cef22f56e2172/CIFAR10/preact_resnet.py#L92
class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def PreActResNet18():
    return PreActResNet(PreActBlock, [2, 2, 2, 2])


def ResNet18(activation_func=nn.ReLU(), num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, activation_fn=activation_func)


def ResNet34(activation_func=nn.ReLU(), num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, activation_fn=activation_func)


def ResNet50(activation_func=nn.ReLU(), num_classes=10):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, activation_fn=activation_func)


def ResNet101(activation_func=nn.ReLU(), num_classes=10):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, activation_fn=activation_func)


def ResNet152(activation_func=nn.ReLU(), num_classes=10):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, activation_fn=activation_func)


def WRN_16_8(dropout=0.1, activation_fn=nn.ReLU, num_classes=10):
    return WideResNet(width_factor=8, depth=16, dropout=dropout, in_channels=3, labels=num_classes, activation_fn=activation_fn)


def WRN_28_10(dropout=0.1, activation_fn=nn.ReLU, num_classes=10):
    return WideResNet(width_factor=10, depth=28, dropout=dropout, in_channels=3, labels=num_classes, activation_fn=activation_fn)


def WRN_34_10(dropout=0.1, activation_fn=nn.ReLU, num_classes=10):
    return WideResNet(width_factor=10, depth=34, dropout=dropout, in_channels=3, labels=num_classes, activation_fn=activation_fn)


def WRN_70_16(dropout=0.1, activation_fn=nn.ReLU, num_classes=10):
    return WideResNet(width_factor=16, depth=70, dropout=dropout, in_channels=3, labels=num_classes, activation_fn=activation_fn)


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())
