from torchvision.transforms import RandomCrop
from torchvision.transforms.functional import pad
import torch

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
        
        
        