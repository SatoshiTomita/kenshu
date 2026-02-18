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
                 save_path,
                 device):
        
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epoch = epoch
        self.save_path = save_path
        self.device=device

    def train_model(self):
        train_loss_log = []
        validation_loss_log = []
        loss_min = 1.0e+10

        for e in tqdm(range(self.epoch)):
            # 引数にdeviceを追加
            minibatch_train_loss = train_loop(self.model, self.train_dataloader, self.optimizer, self.loss_fn,self.device)
            train_loss_log.append(minibatch_train_loss)

            # 修正1-9：deviceを引数に追加
            minibatch_val_loss = validation_loop(self.model, self.val_dataloader, self.loss_fn,self.device)
            validation_loss_log.append(minibatch_val_loss)
            
            wandb.log({"train_loss": minibatch_train_loss,
            "validation_loss": minibatch_val_loss,
            "epoch": e,
            })
            if loss_min > minibatch_val_loss:
                save_file(self.model.state_dict(), f'{self.save_path}/best_model.safetensors')
                loss_min = minibatch_val_loss

        return train_loss_log, validation_loss_log

# 修正1-9 deviceを引数に追加
def train_loop(model, train_dataloader, optimizer, loss_fn,device):
    model.train()
    minibatch_train_loss = 0
    for i, batch in enumerate(train_dataloader):
        input, target = batch
        # 修正1-9　batchからinputとtargetに分割したデータを専用メモリへとコピーする
        input,target=input.to(device),target.to(device)
        prediction = model(input)
        train_loss = loss_fn(prediction, target)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        minibatch_train_loss += train_loss.item()

    minibatch_train_loss /= (i+1)
    return minibatch_train_loss

# 修正1-9 deviceを引数に追加
def validation_loop(model, val_dataloader, loss_fn,device):
    model.eval()
    minibatch_val_loss = 0
    with torch.no_grad():
        for j, batch in enumerate(val_dataloader):
            input, target = batch
            # validation時もdeviceへ送る様にする
            input,target=input.to(device),target.to(device)
            prediction = model(input)
            validation_loss = loss_fn(prediction, target)

            minibatch_val_loss += validation_loss.item()

    minibatch_val_loss /= (j+1)
    return minibatch_val_loss