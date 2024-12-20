import torch
import torch.nn as nn
import torch.nn.functional as F


def squash(input, eps=10e-21):
    n = torch.norm(input, dim=-1, keepdim=True)
    return (1 - 1 / (torch.exp(n) + eps)) * (input / (n + eps))


def length(input):
    return torch.sqrt(torch.sum(input ** 2, dim=-1) + 1e-8)


def mask(input):
    if type(input) is list:
        input, mask = input
    else:
        x = torch.sqrt(torch.sum(input ** 2, dim=-1))
        mask = F.one_hot(torch.argmax(x, dim=1), num_classes=x.shape[1]).float()

    masked = input * mask.unsqueeze(-1)
    return masked.view(input.shape[0], -1)


class PrimaryCapsLayer(nn.Module):
    def __init__(self, in_channels, kernel_size, num_capsules, dim_capsules, stride=1):
        super(PrimaryCapsLayer, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
            stride=stride, groups=in_channels, padding="valid", )
        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules

    def forward(self, input):
        output = self.depthwise_conv(input)
        output = output.view(output.size(0), self.num_capsules, self.dim_capsules)
        return squash(output)


class RoutingLayer(nn.Module):
    def __init__(self, num_capsules, dim_capsules):
        super(RoutingLayer, self).__init__()
        self.W = nn.Parameter(torch.Tensor(num_capsules, 16, 8, dim_capsules))
        self.b = nn.Parameter(torch.zeros(num_capsules, 16, 1))
        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.W)
        nn.init.zeros_(self.b)

    def forward(self, input):
        u = torch.einsum("...ji,kjiz->...kjz", input, self.W)
        c = torch.einsum("...ij,...kj->...i", u, u)[..., None]
        c = c / torch.sqrt(torch.Tensor([self.dim_capsules]).type(torch.FloatTensor))
        c = torch.softmax(c, axis=1)
        c = c + self.b
        s = torch.sum(torch.mul(u, c), dim=-2)
        return squash(s)


class EfficientCapsNet(nn.Module):
    def __init__(self, rout_capsules, route_dim):
        super(EfficientCapsNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding="valid")
        self.batch_norm1 = nn.BatchNorm2d(num_features=32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding="valid")
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, padding="valid")
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, 3, stride=2, padding="valid")
        self.batch_norm4 = nn.BatchNorm2d(128)

        self.primary_caps = PrimaryCapsLayer(in_channels=128, kernel_size=9, num_capsules=16, dim_capsules=8)
        self.digit_caps = RoutingLayer(num_capsules=rout_capsules, dim_capsules=route_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.conv4.weight)

    def forward(self, x):
        x = torch.relu(self.batch_norm1(self.conv1(x)))
        x = torch.relu(self.batch_norm2(self.conv2(x)))
        x = torch.relu(self.batch_norm3(self.conv3(x)))
        x = torch.relu(self.batch_norm4(self.conv4(x)))
        x = self.primary_caps(x)
        x = self.digit_caps(x)
        return x
