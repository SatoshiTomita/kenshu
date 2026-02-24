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
            
            wandb.log({"train_total_loss": minibatch_train_loss["loss"],
            "train_recon_loss": minibatch_train_loss["reconstruction"],
            "train_kld_loss": minibatch_train_loss["kld"],
            "validation_total_loss": minibatch_val_loss["loss"],
            "validation_recon_loss": minibatch_val_loss["reconstruction"],
            "validation_kld_loss": minibatch_val_loss["kld"],
            "epoch": e,
            })
            if loss_min > minibatch_val_loss["loss"]:
                save_file(self.model.state_dict(), f'{self.save_path}/best_model.safetensors')
                loss_min = minibatch_val_loss["loss"]

        return train_loss_log, validation_loss_log

def train_loop(model, train_dataloader, optimizer, loss_fn, device):
    model.train()
    minibatch_train_loss = 0
    minibatch_train_recon_loss = 0
    minibatch_train_kld = 0
    for  i, (input, _) in enumerate(train_dataloader):
          input = input.to(device)

          prediction = model(input)

          train_loss, train_recon_loss, train_kld = loss_fn(prediction, input)

          optimizer.zero_grad()
          train_loss.backward()
          optimizer.step()

          train_loss = train_loss.detach().clone().cpu()
          train_loss = train_loss.numpy()
          minibatch_train_loss += train_loss

          train_recon_loss = train_recon_loss.detach().clone().cpu()
          train_recon_loss = train_recon_loss.numpy()
          minibatch_train_recon_loss += train_recon_loss

          train_kld = train_kld.detach().clone().cpu()
          train_kld = train_kld.numpy()
          minibatch_train_kld += train_kld

    minibatch_train_loss /= (i+1)
    minibatch_train_recon_loss /= (i+1)
    minibatch_train_kld /= (i+1)

    train_loss_dict = {}
    train_loss_dict["loss"] = minibatch_train_loss
    train_loss_dict["reconstruction"] = minibatch_train_recon_loss
    train_loss_dict["kld"] = minibatch_train_kld
    
    return train_loss_dict
# 修正1-9 deviceを引数に追加
def validation_loop(model, val_dataloader, loss_fn,device):
    model.eval()

    total_loss_sum = 0
    recon_loss_sum = 0
    kld_sum = 0
    num_batches = 0

    with torch.no_grad():
        for input, _ in val_dataloader:
            input = input.to(device)

            total_loss, recon_loss, kld = loss_fn(model(input), input)

            total_loss_sum += total_loss.item()
            recon_loss_sum += recon_loss.item()
            kld_sum += kld.item()
            num_batches += 1

    return {
        "loss": total_loss_sum / num_batches,
        "reconstruction": recon_loss_sum / num_batches,
        "kld": kld_sum / num_batches,
    }