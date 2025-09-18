from PIL import Image
import os
from torch.utils.data import Dataset
from typing import Optional, Callable


class ROCCOV2Dataset(Dataset):
    def __init__(
        self,
        data_path: str,
        split: str,
        transform: Optional[Callable] = None,
        tokenizer: Optional[Callable] = None,
        return_multimodal_format: bool = False
    ):
        """
        ROCCO v2 Dataset loader for FuseLIP
        Args:
            data_path: Path to the ROCOv2_data directory
            split: One of 'train', 'validation', or 'test'
            transform: Optional image transform
            tokenizer: Optional text tokenizer
            return_multimodal_format: If True, returns format compatible with multimodal training
        """
        self.data_path = os.path.join(data_path, "ROCOv2_data", split)
        self.transform = transform
        self.tokenizer = tokenizer
        self.return_multimodal_format = return_multimodal_format
        
        # Get all image files
        self.image_files = [f for f in os.listdir(self.data_path) if f.endswith('.jpg')]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_path = os.path.join(self.data_path, img_file)
        txt_path = img_path + '.txt'
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
            
        # Load caption
        with open(txt_path, 'r', encoding='utf-8') as f:
            txt = f.read().strip()
            
        if self.tokenizer:
            txt = self.tokenizer([str(txt)])[0]
            
        if self.return_multimodal_format:
            return self.tokenizer("")[0], img, txt, None
            
        return img, txt


def get_roccov2_dataset(
    data_path: str,
    split: str,
    transform: Optional[Callable] = None,
    tokenizer: Optional[Callable] = None,
    return_multimodal_format: bool = False
):
    """Helper function to create ROCCOV2Dataset instance"""
    return ROCCOV2Dataset(
        data_path=data_path,
        split=split,
        transform=transform,
        tokenizer=tokenizer,
        return_multimodal_format=return_multimodal_format
    )