import torch
import numpy as np

class Predict():
    def __init__(self,
                 cfg,
                 model,
                 device,all_data): 
        self.model = model
        self.cfg = cfg
        # deviceの保存
        self.device=device  
        # 初期データをdeviceへ
        self.data = prepare_data(cfg).to(device)
        self.all_data = torch.tensor(all_data, dtype=torch.float32)
    
    def update_model(self, model):
        self.model = model
    
    def test_loss(self, test_dataloader, loss_fn,device):
        self.model.eval()
        minibatch_test_loss = 0
        with torch.no_grad():
            for j, batch in enumerate(test_dataloader):
                input, target = batch
                # deviceへ転送
                input,target=input.to(device),target.to(device)
                # 修正:1-10
                self.model.initialize(input.shape[0],device)
                prediction = self.model(input)
                test_loss = loss_fn(prediction, target)

                minibatch_test_loss += test_loss.item()

        minibatch_test_loss /= (j+1)
        return minibatch_test_loss
    
    def predict(self):
        predictions_list = []
        
        for i in range(self.cfg.train_data.input_timestep):
            # 修正1-9 numpyする前にcpuを挟む様にする
            predictions_list.append(np.reshape(self.data.detach().clone().cpu().numpy(), [self.cfg.train_data.input_timestep, -1])[i])
            
        with torch.no_grad():
            for _ in range(500):
                prediction = self.model(self.data)
                # numpyする前にcpuを挟む
                predictions_list.append(prediction.detach().clone().cpu().numpy())

                self.data = torch.cat([self.data, prediction], axis=0)[2:]

        return np.stack(predictions_list)
    
    # 学習時：軌道の時系列データ（正解データ）を入力として用いる
    # テスト時：１つ前の自分の予測結果を次のステップの入力として扱う
    def predict_withRNN(self):
        predictions_list = []
        self.model.eval()
        
        # 1. データの形状を整える（以前の修正のまま）
        if self.all_data.dim() == 3:
            initial_source = self.all_data[0:1, :, :]
        else:
            initial_source = self.all_data.unsqueeze(0)

        input_step = self.cfg.train_data.input_timestep
        current_input = initial_source[:, :input_step, :].to(self.device)
        
        for i in range(input_step):
            predictions_list.append(current_input[0, i].cpu().numpy())
            
        with torch.no_grad():
            self.model.initialize(1, self.device)
            for _ in range(150):
                # 2. 予測。modelの戻り値は [1, 2] (2次元)
                prediction = self.model(current_input) 
                
                # 3. リストに追加
                predictions_list.append(prediction[0].cpu().numpy())
                
                # 4. ★ここを修正！ スライディングウィンドウの更新
                # prediction [1, 2] を [1, 1, 2] に拡張してから cat する
                next_point = prediction.unsqueeze(1) 
                current_input = torch.cat([current_input[:, 1:, :], next_point], dim=1)

        return np.stack(predictions_list)
def prepare_data(cfg):
    theta = np.linspace(cfg.gen_data.min, cfg.gen_data.max, cfg.gen_data.data_length)

    x = np.cos(theta)
    y = np.sin(theta*2)

    data = np.stack([x, y], axis=1)
    # 修正1:10最初の1点を取り出す
    first_point=data[0]
    # 修正1-10:RNN用の[Batch,Seq,Dim]に整形する
    rnn_input=first_point.reshape(1,1,2)

    return torch.Tensor(rnn_input).float()




        