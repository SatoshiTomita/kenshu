from torch import nn

class FNN(nn.Module):
    """
        Example NN \n
        input : input
        output : output
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # モデルのレイヤー定義
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    # モデルが実際に使われる際の処理
    def forward(self, input):
        # 入ってきたデータ(input)を準備したネットワークに通す
        output = self.net(input)
        return output
    
