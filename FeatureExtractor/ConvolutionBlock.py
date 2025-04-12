import torch
class ConvBlock(torch.nn.Module):
    def __init__(self,in_channels:int,out_channels:int,momentum:float=0.9,kernel_size:int=3):
        super(ConvBlock,self).__init__()
        self.conv_1=torch.nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,
                                    padding="same")
        self.conv_2=torch.nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=kernel_size,
                                    padding="same")
        self.bn=torch.nn.BatchNorm2d(num_features=out_channels,momentum=momentum)
        self.max_pool=torch.nn.MaxPool2d(kernel_size=2,stride=2)
        self.activation=torch.nn.ReLU(inplace=True)
    def forward(self,X:torch.Tensor)->torch.Tensor:
        X=self.conv_1(X)
        X=self.conv_2(X)
        X=self.activation(X)
        X=self.bn(X)
        X=self.max_pool(X)
        return X