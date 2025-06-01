import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from pathlib import Path


class SkinCancerDataset(Dataset):
    def __init__(self, dataframe, img_dirs, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.img_dirs = img_dirs  # List of image directory paths
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row['image_id']
        label = row['binary_target']

        # Find the image in one of the directories
        image_path = self._find_image_path(image_id)

        # Load and convert image
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

    def _find_image_path(self, image_id):
        """Find image in one of the possible directories"""
        for img_dir in self.img_dirs:
            img_path = Path(f"{img_dir}/{image_id}.jpg")
            if img_path.exists():
                return img_path
        raise FileNotFoundError(f"Image {image_id}.jpg not found in any directory")


# Test your dataset
if __name__ == "__main__":
    df = pd.read_csv('data/binary_metadata.csv')
    img_dirs = ['data/HAM10000_images_part_1', 'data/HAM10000_images_part_2']

    dataset = SkinCancerDataset(df.head(5), img_dirs)
    print(f"Dataset size: {len(dataset)}")

    # Test loading one sample
    image, label = dataset[0]
    print(f"Image size: {image.size}, Label: {label}")
