from tqdm import tqdm
import torch
import wandb
from safetensors.torch import save_file

class Trainer():
    def __init__(self,
                 model,
                 train_dataloader,
                 val_dataloader,
                 optimizer,
                 loss_fn,
                 epoch,
                 save_path):
        
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epoch = epoch
        self.save_path = save_path

    def train_model(self):
        train_loss_log = []
        validation_loss_log = []
        loss_min = 1.0e+10

        for e in tqdm(range(self.epoch)):
            minibatch_train_loss = train_loop(self.model, self.train_dataloader, self.optimizer, self.loss_fn)
            train_loss_log.append(minibatch_train_loss)

            minibatch_val_loss = validation_loop(self.model, self.val_dataloader, self.loss_fn)
            validation_loss_log.append(minibatch_val_loss)
            # wandbでグラフ化
            wandb.log({"train_loss": minibatch_train_loss,
            "validation_loss": minibatch_val_loss,
            "epoch": e,
            })
            if loss_min > minibatch_val_loss:
                save_file(self.model.state_dict(), f'{self.save_path}/best_model.safetensors')
                loss_min = minibatch_val_loss

        return train_loss_log, validation_loss_log


def train_loop(model, train_dataloader, optimizer, loss_fn):
    # trainループの初期化
    model.train()
    minibatch_train_loss = 0
    # dataloaderからデータを取り出す
    for i, batch in enumerate(train_dataloader):
        input, target = batch
        # inputをモデルに入れる
        prediction = model(input)
        # モデルの出力と正解を損失関数に入れる
        train_loss = loss_fn(prediction, target)

        # 勾配をリセット
        optimizer.zero_grad()
        # 改めて勾配を計算
        train_loss.backward()
        # パラメータ更新
        optimizer.step()

        # 損失の和を計算
        minibatch_train_loss += train_loss.item()
    # index分で損失を割り平均的な損失を算出する
    minibatch_train_loss /= (i+1)
    return minibatch_train_loss

def validation_loop(model, val_dataloader, loss_fn):
    # model.evelでモードを変更
    model.eval()
    minibatch_val_loss = 0
    with torch.no_grad():
        for j, batch in enumerate(val_dataloader):
            # パラメータの更新や勾配計算んを行わない
            input, target = batch
            prediction = model(input)
            validation_loss = loss_fn(prediction, target)

            minibatch_val_loss += validation_loss.item()

    minibatch_val_loss /= (j+1)
    return minibatch_val_loss