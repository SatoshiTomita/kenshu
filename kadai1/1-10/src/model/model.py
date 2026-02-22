import torch
from torch import nn

# class FNN(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super().__init__()

#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, output_dim)
#         )

#     def forward(self, input):
#         output = self.net(input)
#         return output

class RNN(nn.Module):
  # ネットワークの定義
  def __init__(self,input_dim,hidden_dim,output_dim):
    super().__init__()
    # batch_first=trueの場合データの形を(Batch,Sequence,Dimension)
    # batch_first=falseの場合データの形を(Sequence,Batch,Dimension)として扱う
    self.rnn=nn.RNN(input_dim,hidden_dim,batch_first=True)
    self.linear=nn.Linear(hidden_dim,output_dim)
    self.hidden_dim=hidden_dim

  # 隠れ状態の初期化
  def initialize(self,batch_size,device):
    self.hidden=torch.zeros(1,batch_size,self.hidden_dim).to(device)

  
  def forward(self, input):
    out, _ = self.rnn(input, None)
    # 10ステップの出力のうち、最後のステップ [:, -1, :] だけを取り出す
    last_step_out = out[:, -1, :]
    return self.linear(last_step_out)

