from awesome_ssl.models.projection_heads.MLP import MLP
from torch.nn import Linear

projection_heads = {
    "MLP": MLP, 
    "Linear": Linear
}