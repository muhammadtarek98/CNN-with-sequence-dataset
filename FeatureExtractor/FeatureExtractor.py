import torch
from ConvolutionBlock import ConvBlock
import torchinfo
class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super(FeatureExtractor,self).__init__()
        self.block1=ConvBlock(in_channels=3,out_channels=64,kernel_size=3)
        self.block2=ConvBlock(in_channels=64,out_channels=128,kernel_size=3)
        self.block3=ConvBlock(in_channels=128,out_channels=256,kernel_size=3)
        self.global_max_pool=torch.nn.AdaptiveMaxPool2d(output_size=(1,1))
    def forward(self,X:torch.Tensor)->torch.Tensor:
        X=self.block1(X)
        X=self.block2(X)
        X=self.block3(X)
        X=self.global_max_pool(X)
        return X
model=FeatureExtractor()
x=torch.randn((1,3,244,244))
torchinfo.summary(model,input_data=x)
