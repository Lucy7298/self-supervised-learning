from torch.utils.data import Dataset
import pickle


class DummyDataset(Dataset): 
    def __init__(self, path): 
        super().__init__()
        with open(path, 'rb') as f: 
            self.data = pickle.load(f)

    def __getitem__(self, idx): 
        rep = self.data['representations'][idx]
        proj = self.data['projections'][idx]
        y = self.data['ys'][idx]
        return (rep, proj), y

    def __len__(self): 
        return self.data['representations'].shape[0]


class DummyModel: 

    def get_representation(X): 
        rep, _ = X 
        return rep 

    def get_projection(X): 
        _, proj = X 
        return proj