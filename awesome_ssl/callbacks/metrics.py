from random import Random
from pytorch_lightning.callbacks import Callback
import torchvision.transforms as T 
from awesome_ssl.augmentations.crop_and_shift import RandomCenterCrop
import torch
from torchmetrics import MeanMetric
from collections import defaultdict
import pandas as pd 
import pickle 

def default_entry(): 
    return MeanMetric(nan_strategy='error')

class InvarianceMetric(Callback): 
    def __init__(self): 
        super().__init__()
        self.reset_state()
        self.transform_invars = {
            "crop": T.RandomResizedCrop(224), 
            "rescale_0.5": RandomCenterCrop(0.5, 0.5), 
            "rescale_2": RandomCenterCrop(2, 2),
            "colorjitter_.4_.4_.4_.2": T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2), 
            "colorjitter_.8_.8_.8_.2": T.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2), 
        }

    def initialize_examples(self, dataloader): 
        return 

    def reset_state(self): 
        self.metrics = {}
        self.moment = {"rep_1": 0, "proj_1": 0, "rep_2": 0, "proj_2": 0}
        self.samples = 0 

    def assign_entry(self, data_dict, data, embed_type, data_type, transform): 
        dict_key = f"{embed_type}_{data_type}_{transform}"
        new_value = data_dict.get(dict_key, [])
        new_value.append(data)
        data_dict[dict_key] = new_value

    @torch.no_grad()
    def calculate_invariance(self, pl_module, batch): 
        X, y = batch
        num_samples = X.shape[0]
        rep = pl_module.get_representation(X)
        proj = pl_module.get_projection(X)
        for transform_name, trans in self.transform_invars.items(): 
            # transform batch 
            t_batch = trans(X)
            # get representations 
            t_rep = pl_module.get_representation(t_batch)
            # get projections 
            t_proj = pl_module.get_projection(t_batch)

            # calculate distance between representations 
            rep_dist = torch.einsum("ij->i", (rep - t_rep)**2).cpu()
            proj_dist = torch.einsum("ij->i", (proj - t_proj)**2).cpu()
            self.assign_entry(self.metrics, rep_dist, "rep", "dist", transform_name)
            self.assign_entry(self.metrics, proj_dist, "proj", "dist", transform_name)

            # calculate dot product between representations 
            rep_sim = torch.nn.functional.cosine_similarity(rep, t_rep).cpu()
            proj_sim = torch.nn.functional.cosine_similarity(proj, t_proj).cpu()
            self.assign_entry(self.metrics, rep_sim, "rep", "sim", transform_name)
            self.assign_entry(self.metrics, proj_sim, "proj", "sim", transform_name)

        # compute first and second moments 
        self.moment["rep_1"] += rep.sum(dim=0).cpu().double()
        self.moment["rep_2"] += (rep**2).sum(dim=0).cpu().double()
        self.moment["proj_1"] += proj.sum(dim=0).cpu().double()
        self.moment["proj_2"] += (proj**2).sum(dim=0).cpu().double()

        self.samples += num_samples

    def on_validation_start(self, trainer, pl_module): 
        self.transform_invars["training_transform"] = pl_module.transform_1

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.calculate_invariance(pl_module, batch)

    def save_data(self, output_path, additional_data): 
        for key, value in self.moment.items(): 
            self.moment[key] = value / self.samples 
        
        transforms_log = {}
        for key, value in self.transform_invars.items(): 
            transforms_log[key] = str(value)

        for key, value in self.metrics.items(): 
            self.metrics[key] = torch.cat(value, axis=0)
        metrics_df = pd.DataFrame.from_dict(self.metrics)
        data = {"metrics": metrics_df, 
                "moments": self.moment, 
                "transforms": transforms_log}
        data.update(additional_data)
        with open(output_path, 'wb') as handle: 
            pickle.dump(data, handle)

        self.reset_state()