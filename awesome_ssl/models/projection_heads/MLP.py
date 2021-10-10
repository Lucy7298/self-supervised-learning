import torch 

class MLP(torch.nn.Module): 
    def __init__(self, input_dim, hidden_dim, output_dim, batchnorm=True): 
        super().__init__()
        self.hidden_layer = torch.nn.Linear(input_dim, hidden_dim)
        if batchnorm: 
            self.bn = torch.nn.BatchNorm1d(hidden_dim)
        else: 
            self.bn = None
        self.relu = torch.nn.ReLU()
        self.output_layer = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x): 
        x = self.hidden_layer(x)
        if self.bn is not None: 
            x = self.bn(x)
        x = self.relu(x)
        return self.output_layer(x)