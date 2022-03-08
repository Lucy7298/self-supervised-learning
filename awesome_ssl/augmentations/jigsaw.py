import torch 
from torch import nn 
from einops import rearrange

class JigsawAugmentation(nn.Module): 
    # x tiles and y tiles must perfectly divide the image dimension
    def __init__(self, x_tiles: int, y_tiles: int):
        super().__init__()
        self.x_tiles = x_tiles 
        self.y_tiles = y_tiles

    def forward(self, image): 
        B, _, H, W = image.shape
        assert H % self.x_tiles == 0 
        assert H % self.y_tiles == 0 

        # each image is split into subgrids
        grids = rearrange(image, 'b c (hs h) (ws w) -> b (hs ws) c h w', hs=self.y_tiles, ws=self.x_tiles)

        # generate shuffle indices 
        indices = torch.argsort(torch.rand((B, self.x_tiles * self.y_tiles)), dim=-1)
        # shuffle the jigsaw 
        grids = grids[torch.arange(B).unsqueeze(-1),indices, :, :, :]
        grids = rearrange(grids, 'b (hs ws) c h w -> b c (hs h) (ws w)', hs=self.y_tiles, ws=self.x_tiles)
        return grids 

    def extra_repr(self) -> str:
        return f"x_tiles={self.x_tiles}, y_tiles={self.y_tiles}"