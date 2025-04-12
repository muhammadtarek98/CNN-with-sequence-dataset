import torch
from FeatureExtractor.FeatureExtractor import  FeatureExtractor
from SequentailModel.FullyConnectedModule import FullyConnectedModule
from SequentailModel.TimeDistributed import TimeDistributed
class CNNSeqModel(torch.nn.Module):
    def __init__(self,num_classes:int,**kwargs):
        self.feature_extractor=FeatureExtractor()
        self.time_distributed=TimeDistributed()
        self.lstm=torch.nn.LSTM()
        self.fc_block=FullyConnectedModule(in_features=64,out_features=1024,dropout=0.5)
        self.fc_block = FullyConnectedModule(in_features=1024,
                                             out_features=512, dropout=0.5)
        self.fc_block = FullyConnectedModule(in_features=512,
                                             out_features=64)
        self.classifier=torch.nn.Linear(in_features=64,out_features=num_classes)
