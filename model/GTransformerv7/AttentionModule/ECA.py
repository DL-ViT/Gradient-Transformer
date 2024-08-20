import torch
from torch import nn
from torch.nn.parameter import Parameter
 
class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.fc1 = nn.Conv2d(in_channels=channel, out_channels=channel//4, kernel_size=1, stride=1, bias=True)
        self.fc2 = nn.Conv2d(in_channels=channel//4, out_channels=channel, kernel_size=1, stride=1, bias=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        global_attention = self.fc1(self.max_pool(x))
        global_attention = self.fc2(global_attention)
        y =self.sigmoid(global_attention+y)
        return  y
class eca_layer_fuse(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer_fuse, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.fc1 = nn.Conv2d(in_channels=channel, out_channels=channel//4, kernel_size=1, stride=1, bias=True)
        self.fc2 = nn.Conv2d(in_channels=channel//4, out_channels=channel, kernel_size=1, stride=1, bias=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self,low,high):
        y = self.avg_pool(high)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        global_attention = self.fc1(self.max_pool(low))
        global_attention = self.fc2(global_attention)
        y =self.sigmoid(global_attention+y)
        return  y