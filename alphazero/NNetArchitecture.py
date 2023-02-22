import torch.nn.functional as F
import torch.nn as nn
import torch

from alphazero.Game import GameState
from alphazero.utils import dotdict
import numpy as np

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
    input_size: int,
    layer_sizes: list,
    output_size: int,
    output_activation=nn.Identity,
    activation=nn.ELU,
):
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)

def temp_softmax(INPUT, t=1.0):
    return torch.exp(F.log_softmax(INPUT / t, dim=1))

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


class ResNet(nn.Module):
    def __init__(self, game_cls: GameState, args: dotdict):
        super(ResNet, self).__init__()
        # game params
        self.channels, self.board_x, self.board_y = game_cls.observation_size()
        self.action_size = game_cls.action_size()
        self.pst = args.pst
        self.conv1 = conv3x3(self.channels, args.num_channels)
        self.bn1 = nn.BatchNorm2d(args.num_channels)

        self.res_layers = []
        for _ in range(args.depth):
            self.res_layers.append(
                ResidualBlock(args.num_channels, args.num_channels)
            )
        self.resnet = nn.Sequential(*self.res_layers)

        self.v_conv = conv1x1(args.num_channels, args.value_head_channels)
        self.v_bn = nn.BatchNorm2d(args.value_head_channels)
        self.v_fc = mlp(
            self.board_x*self.board_y*args.value_head_channels,
            args.value_dense_layers,
            game_cls.num_players() + game_cls.has_draw(),
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

        #return F.log_softmax(pi / 1.3, dim=1), F.log_softmax(v, dim=1)
        return F.log_softmax(pi, dim=1), F.log_softmax(v, dim=1)


class FullyConnected(nn.Module):
    """
    Fully connected network which operates in the same way as NNetArchitecture.
    The fully_connected function is used to create the network, as well as the
    policy and value heads. Forward method returns log_softmax of policy and value head.
    """
    def __init__(self, game_cls: GameState, args: dotdict):
        super(FullyConnected, self).__init__()
        # get input size
        self.input_size = np.prod(game_cls.observation_size())

        self.input_fc = mlp(
            self.input_size,
            args.input_fc_layers,
            args.input_fc_layers[-1],
            activation=nn.ReLU
        )
        self.v_fc = mlp(
            args.input_fc_layers[-1],
            args.value_dense_layers,
            game_cls.num_players() + game_cls.has_draw(),
            activation=nn.Identity
        )
        self.pi_fc = mlp(
            args.input_fc_layers[-1],
            args.policy_dense_layers,
            game_cls.action_size(),
            activation=nn.Identity
        )

    def forward(self, s):
        # s: batch_size x num_channels x board_x x board_y
        # reshape s for input_fc
        s = s.view(-1, self.input_size)
        s = self.input_fc(s)
        v = self.v_fc(s)
        pi = self.pi_fc(s)

        return F.log_softmax(pi, dim=1), F.log_softmax(v, dim=1)
