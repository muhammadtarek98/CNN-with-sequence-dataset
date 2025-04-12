import torch
class TimeDistributed(torch.nn.Module):
    def __init__(self,module:torch.nn.Module):
        super(TimeDistributed,self).__init__()
        self.module=module
    def forward(self,x:torch.Tensor)->torch.Tensor:
        x_reshape=x.contiguous().view(-1,x.size(-1))
        module_output=self.module(x_reshape)
        x_reshape=module_output.contiguous.view(x.size(0),-1,module_output.size(-1))
        return x_reshape

