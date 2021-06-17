import torch.nn.functional as F
import torch.nn as nn
import torch


# 1x1 convolution
def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=False)

# 3*3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)

# fully connected layers
def mlp(
    input_size,
    layer_sizes,
    output_size,
    output_activation=nn.Identity,
    activation=nn.ELU,
):
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResidualBlock, self).__init__()
        stride = 1
        if downsample:
            stride = 2
            self.conv_ds = conv1x1(in_channels, out_channels, stride)
            self.bn_ds = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)

    def forward(self, x):
        residual = x
        out = x
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample:
            residual = self.conv_ds(x)
            residual = self.bn_ds(residual)
        out += residual
        return out


class NNetArchitecture(nn.Module):
    def __init__(self, game_cls, args):
        super(NNetArchitecture, self).__init__()
        # game params
        self.channels, self.board_x, self.board_y = game_cls.observation_size()
        self.action_size = game_cls.action_size()

        self.conv1 = conv3x3(self.channels, args.num_channels)
        self.bn1 = nn.BatchNorm2d(args.num_channels)

        self.res_layers = []
        for _ in range(args.depth):
            self.res_layers.append(ResidualBlock(
                args.num_channels, args.num_channels))
        self.resnet = nn.Sequential(*self.res_layers)

        self.v_conv = conv1x1(args.num_channels, args.value_head_channels)
        self.v_bn = nn.BatchNorm2d(args.value_head_channels)
        self.v_fc = mlp(
            self.board_x*self.board_y*args.value_head_channels,
            args.value_dense_layers,
            1,
            activation=nn.Identity
        )

        self.pi_conv = conv1x1(args.num_channels, args.policy_head_channels)
        self.pi_bn = nn.BatchNorm2d(args.policy_head_channels)
        self.pi_fc = mlp(
            self.board_x*self.board_y*args.policy_head_channels,
            args.policy_dense_layers,
            self.action_size,
            activation=nn.Identity
        )

    def forward(self, s):
        # s: batch_size x num_channels x board_x x board_y
        s = s.view(-1, self.channels, self.board_x, self.board_y)
        s = F.relu(self.bn1(self.conv1(s)))
        s = self.resnet(s)

        v = self.v_conv(s)
        v = self.v_bn(v)
        v = torch.flatten(v, 1)
        v = self.v_fc(v)

        pi = self.pi_conv(s)
        pi = self.pi_bn(pi)
        pi = torch.flatten(pi, 1)
        pi = self.pi_fc(pi)

        return F.log_softmax(pi, dim=1), torch.tanh(v)
