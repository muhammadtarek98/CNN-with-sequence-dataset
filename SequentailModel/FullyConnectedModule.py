import torch
class FullyConnectedModule(torch.nn.Module):
    def __init__(self,in_features:int,out_features:int,dropout:float=None):
        super(FullyConnectedModule,self).__init__()
        self.fc=torch.nn.Linear(in_features=in_features,out_features=out_features)
        self.activation = torch.nn.ReLU(inplace=True)
        if dropout:
            self.use_dropout=True
            self.dropout=torch.nn.Dropout(inplace=True,p=dropout)
        else:
            self.use_dropout=False
    def forward(self,X:torch.Tensor)->torch.Tensor:
        X=self.fc(X)
        X=self.activation(X)
        if self.use_dropout:
            X=self.dropout(X)
        return X
