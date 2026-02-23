import torchvision.transforms as transforms


class ObsTransform:
    def __init__(self):
        self.obs_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
        ])