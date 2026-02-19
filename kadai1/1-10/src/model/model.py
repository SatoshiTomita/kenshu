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

  
  def forward(self,input):
    # self.rnnに入力データ(input)と前のステップの記憶のself.hiddenを渡し各ステップの特徴量であるoutと更新された記憶のhiddenを受け取る
    out,hidden=self.rnn(input,self.hidden)
    # rnnが計算した特徴量をlinear層に通して最終的な予測値に変換する
    output=self.linear(out)
    # 新しく更新された記憶をクラス変数のself.hiddenに代入して隠れ状態を引き継ぐ
    self.hidden=hidden
    return output

  