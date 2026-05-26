import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.vision_transformer import Block
from timm.models.layers import trunc_normal_
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
from scipy.spatial.transform import Rotation as R

def quaternion_angle_loss(q_pred, q_true):
    q_pred = q_pred / torch.norm(q_pred, dim=1, keepdim=True)
    q_true = q_true / torch.norm(q_true, dim=1, keepdim=True)
    dot_product = torch.sum(q_pred * q_true, dim=1)
    dot_product = torch.clamp(dot_product, min=-1.0 + 1e-7, max=1.0 - 1e-7)
    
    angle_difference = 2 * torch.acos(torch.abs(dot_product))
    return angle_difference.mean()


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    

class ConvPoolingPolicyHead(nn.Module):
    def __init__(self, input_channels, 
                 conv1_out_channels=128, 
                 conv2_out_channels=256, 
                 conv3_out_channels=512,
                 kernel_size=3, padding=1, use_se=False):
        super(ConvPoolingPolicyHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, 
                               out_channels=conv1_out_channels, 
                               kernel_size=kernel_size, 
                               padding=padding)
        self.ln1 = nn.LayerNorm(conv1_out_channels) 
        self.conv2 = nn.Conv2d(in_channels=conv1_out_channels, 
                               out_channels=conv2_out_channels, 
                               kernel_size=kernel_size, 
                               padding=padding)
        self.ln2 = nn.LayerNorm(conv2_out_channels)
        self.conv3 = nn.Conv2d(in_channels=conv2_out_channels, 
                               out_channels=conv3_out_channels, 
                               kernel_size=kernel_size, 
                               padding=padding)
        self.ln3 = nn.LayerNorm(conv3_out_channels)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        if use_se:
            self.se1 = SEBlock(conv1_out_channels)
            self.se2 = SEBlock(conv2_out_channels)
        self.use_se = use_se
        
    def forward(self, x):

        x = self.conv1(x)                        
        x = x.permute(0, 2, 3, 1)                  
        x = self.ln1(x)
        x = x.permute(0, 3, 1, 2)                    
        x = F.leaky_relu(x, negative_slope=0.01)
        if self.use_se:
            x = self.se1(x)
        x = self.conv2(x)                         
        x = x.permute(0, 2, 3, 1)                   
        x = self.ln2(x)
        x = x.permute(0, 3, 1, 2)                    
        x = F.leaky_relu(x, negative_slope=0.01)
        if self.use_se:
            x = self.se2(x)
        x = self.conv3(x)                          
        x = x.permute(0, 2, 3, 1)                    
        x = self.ln3(x)
        x = x.permute(0, 3, 1, 2)                    
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.pool(x)                         
        x = x.view(x.size(0), -1)                    
        return x


class ConvPoolingHead(nn.Module):
    def __init__(self, input_channels, conv1_out_channels=128, conv2_out_channels=256, conv3_out_channels=512,
                 kernel_size=3, padding=1, group_num=4, pool_kernel_size=2, pool_stride=2, pool_padding=0):
        super(ConvPoolingHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=conv1_out_channels, kernel_size=kernel_size, padding=padding)
        self.gn1 = nn.GroupNorm(group_num, conv1_out_channels)
        self.se1 = SEBlock(conv1_out_channels)
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
        self.conv2 = nn.Conv2d(in_channels=conv1_out_channels, out_channels=conv2_out_channels, kernel_size=kernel_size, padding=padding)
        self.gn2 = nn.GroupNorm(group_num, conv2_out_channels)
        self.se2 = SEBlock(conv2_out_channels)
        self.conv3 = nn.Conv2d(in_channels=conv2_out_channels, out_channels=conv3_out_channels, kernel_size=kernel_size, padding=padding)
        self.gn3 = nn.GroupNorm(group_num, conv3_out_channels)
        self.se3 = SEBlock(conv3_out_channels)
        self.pool2 = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.gn1(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.se1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.gn2(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.se2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.gn3(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.se3(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        return x
    
class MlpHead(nn.Module):
    def __init__(self, input_dim, hidden1_dim=256, hidden2_dim=128, output_dim=4, dropout_rate=0.2):
        super(MlpHead, self).__init__()
        self.mlp_layers = nn.Sequential(
            nn.Linear(input_dim, hidden1_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden1_dim, hidden2_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden2_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp_layers(x)
    
class ResNetPerception(nn.Module):
    def __init__(self, num_classes=4, hidden1_dim=256, hidden2_dim=128, dropout_rate=0.2,backbone = 'resnet34', pretrained = False, freeze=False):
        super(ResNetPerception, self).__init__()
        if backbone == 'resnet34':
            self.model = timm.create_model('resnet34', pretrained=pretrained)
            print('Using resnet34')
        elif backbone == 'resnet50':
            self.model = timm.create_model('resnet50', pretrained=pretrained)
            print('Using resnet50')
        elif backbone == 'resnet18':
            self.model = timm.create_model('resnet18', pretrained=pretrained)
            print('Using resnet18')
        # Replace the default fully connected layer with MLPHead
        in_features = self.model.fc.in_features
        self.model.fc = MlpHead(in_features, hidden1_dim=hidden1_dim, hidden2_dim=hidden2_dim, output_dim=num_classes, dropout_rate=dropout_rate)
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            # unfreeze the MLPHead
            for param in self.model.fc.parameters():
                param.requires_grad = True
    
    def forward(self, x):
        return self.model(x)