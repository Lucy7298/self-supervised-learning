from torchvision.transforms import RandomCrop
from torchvision.transforms.functional import resize, center_crop, pad
import torch
import math

class CropAndShift(torch.nn.Module): 
    def __init__(self, crop_size: int, shift: bool): 
        super().__init__()
        self.crop = RandomCrop(crop_size)
        self.crop_size = crop_size
        self.shift = shift
        
    def forward(self, image): 
        _, _, H, W = image.shape
        assert H > self.crop_size 
        assert W > self.crop_size
        
        image = self.crop(image)
        x_remain = W - self.crop_size
        y_remain = H - self.crop_size
        if self.shift: 
            left_pad = torch.randint(x_remain, (1, 1)).item()
            top_pad = torch.randint(y_remain, (1, 1)).item()
        else: 
            left_pad = (x_remain) // 2 
            top_pad = (y_remain) // 2 
            
        return pad(image, padding=[left_pad, top_pad, x_remain - left_pad, y_remain-top_pad])

    def extra_repr(self) -> str:
        return f"crop_size={self.crop_size}, shift={self.shift}"

class RandomCenterCrop(torch.nn.Module): 
    def __init__(self, min_ratio: float, max_ratio: float): 
        super().__init__()
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio 
        assert self.min_ratio <= self.max_ratio 
        
    def forward(self, image): 
        _, _, H, W = image.shape
        min_size = math.floor(H*self.min_ratio)
        max_size = math.ceil(H*self.max_ratio)
        if self.min_ratio < self.max_ratio: 
            new_size = torch.randint(min_size, max_size, (1, 1)).item()
        elif self.min_ratio == self.max_ratio: 
            new_size = min_size

        image = resize(image, new_size)
        image = center_crop(image, (H, W))
        return image

    def extra_repr(self) -> str:
        return f"min_ratio={self.min_ratio}, max_ratio={self.max_ratio}"

        
        