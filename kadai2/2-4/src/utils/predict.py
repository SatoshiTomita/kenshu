import torch
import numpy as np

class Predict():
    def __init__(self,
                 cfg,
                 model,
                 device): 
        self.model = model
        self.cfg = cfg
        # deviceの保存
        self.device=device  
        # 初期データをdeviceへ
        self.data = prepare_data(cfg).to(device)
    
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


def prepare_data(cfg):
    theta = np.linspace(cfg.gen_data.min, cfg.gen_data.max, cfg.gen_data.data_length)

    x = np.cos(theta)
    y = np.sin(theta*2)

    data = np.stack([x, y], axis=1)
    data = data[:cfg.train_data.input_timestep]

    data = torch.Tensor(data).float().view(-1)
    return data