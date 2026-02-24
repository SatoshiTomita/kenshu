import torch
import numpy as np

class Predict():
    def __init__(self,
                 cfg,
                 model):
        self.model = model
        self.cfg = cfg
        self.data = prepare_data(cfg)
    
    def update_model(self, model):
        self.model = model
    
    def test_loss(self, test_dataloader, loss_fn):
        self.model.eval()
        minibatch_test_loss = 0
        with torch.no_grad():
            for j, batch in enumerate(test_dataloader):
                input, target = batch
                prediction = self.model(input)
                test_loss = loss_fn(prediction, target)

                minibatch_test_loss += test_loss.item()

        minibatch_test_loss /= (j+1)
        return minibatch_test_loss
    
    def predict(self):
        predictions_list = []
        # 予測ループ
        for i in range(self.cfg.train_data.input_timestep):
            predictions_list.append(np.reshape(self.data.detach().clone().numpy(), [self.cfg.train_data.input_timestep, -1])[i])
            
        with torch.no_grad():
            for _ in range(500):
                prediction = self.model(self.data)
                # 予測した新しい1点をリストに入れる
                predictions_list.append(prediction.detach().clone().numpy())
                # AIが出した答えpredictionをcatメソッドを使い連結
                self.data = torch.cat([self.data, prediction], axis=0)[2:]

        return np.stack(predictions_list)


def prepare_data(cfg):
    min_value = cfg.gen_data.min
    max_value = cfg.gen_data.max * np.pi
    points_per_round = cfg.gen_data.points_per_round
    num_rounds = cfg.gen_data.num_rounds
    theta = np.linspace(min_value, max_value * num_rounds, points_per_round * num_rounds)

    x = np.cos(theta)
    y = np.sin(theta*2)

    data = np.stack([x, y], axis=1)
    data = data[:cfg.train_data.input_timestep]

    data = torch.Tensor(data).float().view(-1)
    return data