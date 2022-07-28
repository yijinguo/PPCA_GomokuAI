import torch.nn.functional as F
import torch.nn as nn
import torch
import sys

sys.path.append('..')

class NNetArchitecture(nn.Module):
    def __init__(self, game, args):
        super(NNetArchitecture, self).__init__()
        # game params
        self.feat_cnt = args.feat_cnt
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        self.cv = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 5, padding = 2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 5, padding = 2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 5, padding = 2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels = 16, out_channels = 10, kernel_size = 5, padding = 2),
            nn.BatchNorm2d(10),
            nn.ReLU()
        )
        self.linear = nn.Sequential(
            nn.Linear(in_features = 10*9*9, out_features = 64),
            nn.ReLU(),
            nn.Linear(in_features = 64, out_features = 64),
            nn.ReLU(),
            nn.Linear(in_features = 64, out_features = self.action_size)
        )

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                #nn.init.kaiming_normal(m.weight.data)#卷积层参数初始化
            elif isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)

        size = self.feat_cnt * self.board_x * self.board_y
        self.layer2 = nn.Sequential(
            nn.Linear(in_features = size, out_features = self.action_size)
        )


    def forward(self, s):
        # batch_size x feat_cnt x board_x x board_y
        s = s.view(-1, self.feat_cnt, self.board_x, self.board_y)   

        feature = self.cv(s) 
        feature = feature.view(feature.size(0), -1)
        pi = self.linear(feature)  

        s = s.reshape(s.size(0), 243)     
        v = self.layer2(s)

        # Think: What are the advantages of using log_softmax ?
        return F.log_softmax(pi, dim=1), torch.tanh(v)