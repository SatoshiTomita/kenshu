import os

import numpy as np
import torch
from box import Box 
import yaml
from safetensors.torch import load_file
from torchvision import datasets, transforms 
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from src.model. vae import VAE
from src.utils.transform import ObsTransform


class Generate():
    def __init__(self, cfg, device, model, numbers, dataloader):
        self.cfg = cfg 
        self.device = device 
        self.model = model 
        self.numbers = numbers
        self.dataloader = dataloader
    
    def reconstruction(self):
        '''
        再構成画像の生成
        '''
        self.model.to(self.device)
        self.model.eval()
        
        fig = plt.figure(figsize=(20,6))
        for i,image in self.numbers.items():
            
            original_image = image.unsqueeze(1).expand(self.cfg.train_data.batch_size,1,28,28)
            original_image = original_image.to(self.device)
            recon_image, z_dist, mean, std, z_sample= self.model(original_image)

            ax = fig.add_subplot(2, 10, i+1, xticks=[], yticks=[])
            ax.imshow(original_image[0].cpu().detach().numpy().squeeze(), cmap='gray')


            ax = fig.add_subplot(2, 10, i+11, xticks=[], yticks=[])
            ax.imshow(recon_image[0].cpu().detach().numpy().squeeze(), cmap='gray')
        
        
        save_dir = "./output/reconstruction"
        os.makedirs(save_dir, exist_ok=True)  

        save_path = os.path.join(save_dir, "output.png")
        fig.savefig(save_path, dpi=300, bbox_inches="tight") 



    def feature_extraction_PCA(self):
        '''
        PCAで各数字の特徴量を可視化
        '''
        self.model.eval()

        Z = []
        labels = []

        for i,(input, label) in enumerate(self.dataloader):
            input = input.to(self.device)  

            output, z_dist, mean, std, z_sample = self.model(input)

            Z.append(z_sample.cpu().detach().numpy())
            labels.append(label.cpu().detach().numpy())
            
            if i+1==10: break


        Z = np.concatenate(Z, axis=0)  
        labels = np.concatenate(labels, axis=0)  

        fig, ax = plt.subplots(figsize=(10,10))
        points = PCA(n_components=2).fit_transform(Z)

        colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'fuchsia', 'grey', 'olive', 'lightblue']
        

        for p, l in zip(points, labels):
            ax.scatter(p[0], p[1], marker='${}$'.format(l), c=colors[l])
        
        ax.set_title("PCA Visualization of Latent Space", fontsize=14)  
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")


        save_path = "./output/feature_extraction/PCA/output.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f'successfully saved at {save_path}')

        return

    def latent_interpolation(self, digit1=3, digit2=8, steps=10):
        '''
        digit1 --> digit2 への線形補間
        steps: degit1 --> digit2 までのステップ数
        
        '''
        self.model.eval()
        
        image1 = self.numbers[digit1].unsqueeze(0).to(self.device)
        image2 = self.numbers[digit2].unsqueeze(0).to(self.device)
        
        _, _, _, _, z1 = self.model(image1)
        _, _, _, _, z2 = self.model(image2)


        interpolated_images = []
        for alpha in np.linspace(0, 1, steps):
            z_interpolated = (1 - alpha) * z1 + alpha * z2
            generated_image = self.model.decoder(z_interpolated).cpu().detach().numpy().squeeze()
            interpolated_images.append(generated_image)


        fig, axes = plt.subplots(1, steps, figsize=(20, 3))
        for i, img in enumerate(interpolated_images):
            axes[i].imshow(img, cmap="gray")
            axes[i].axis("off")
        
        save_path = f"./output/interpolation/output{digit1}to{digit2}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f'successfully saved at {save_path}')

        return
   
def main():
    with open('./conf/config.yaml', 'r') as yml:
        cfg = Box(yaml.safe_load(yml))
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model_param_path = f"result/{cfg.wandb.train_name}/best_model.safetensors"
    model_param = load_file(model_param_path)
    
    model = VAE(cfg).to(device)
    model.load_state_dict(model_param)
    
    transform = ObsTransform()
    dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    
    dataloader = DataLoader(dataset, batch_size=cfg.train_data.batch_size, shuffle=False)

    numbers = dict() 
    for data, label in dataset:
        numbers[label] = data
    
    generation = Generate(
        cfg=cfg,
        device=device, 
        model=model,
        numbers=numbers,
        dataloader=dataloader 
    )
    
    generation.reconstruction() #画像の再構成
    generation.feature_extraction_PCA() #特徴量抽出(PCA)
    generation.latent_interpolation(digit1=3,digit2=8,steps=10) # digit1 --> digit2 への線形補間
 

if __name__ == '__main__': 
    main()