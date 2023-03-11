import os
from PIL import Image
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.files_A = sorted(os.listdir(os.path.join(root, "train_monet")))
        self.files_B = sorted(os.listdir(os.path.join(root, "train_pictures")))

    def __getitem__(self, index):
        img_A = Image.open(os.path.join("monet2photo", "train_monet", self.files_A[index % len(self.files_A)]))
        img_B = Image.open(os.path.join("monet2photo", "train_pictures", self.files_B[index % len(self.files_B)]))

        if self.transform is not None:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))