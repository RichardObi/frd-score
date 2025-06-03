import os
from PIL import Image
from torch.utils.data import Dataset

class SimpleImageDataset(Dataset):
    def __init__(self, image_folder, image_mode, transform=None):
        self.image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform
        self.image_mode = image_mode
        
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        x = Image.open(image_path).convert(self.image_mode)
        if self.transform is not None:
            x = self.transform(x)
        return x

    def __len__(self):
        return len(self.image_paths)