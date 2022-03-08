from random import Random
from pytorch_lightning.callbacks import Callback
import torchvision.transforms as T 
from awesome_ssl.augmentations.crop_and_shift import RandomCenterCrop
import torch
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchmetrics import MeanMetric
from collections import defaultdict
import pandas as pd 
import pickle 


RELATIVE_DISTANCE_TRANFORM = T.Compose([
        T.Resize(size=224), 
        T.CenterCrop(size=224),
        T.ToTensor(), 
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]), 
    ])

def update_similarity_table(similarity_per_data, y, total_y, device): 
    poss_labels = torch.arange(total_y, device=device).unsqueeze(0)
    y = y.unsqueeze(1).to(device)
    M = (poss_labels == y).type(similarity_per_data.type())
    update = (similarity_per_data.t() @ M)
    return update 



class RelativeDistance(Callback): 
    def __init__(self): 
        super().__init__()
        self.reset_state()

    def reset_state(self): 
        self.rep_confusion_dist = torch.zeros((10, 10))
        self.rep_confusion_sim = torch.zeros((10, 10))
        self.rep_sim = torch.zeros((10, 10))
        self.proj_confusion_dist = torch.zeros((10, 10))
        self.proj_confusion_sim = torch.zeros((10, 10))
        self.proj_sim = torch.zeros((10, 10))
        self.idx_to_label_list = None # this will be set outside
        self.images = None
        self.labels = None
        self.num_classes = None

    def initialize_examples(self, eval_dataloader):
        class_to_idx = eval_dataloader.dataset.class_to_idx 
        dataset = ImageFolder("/mnt/nfs/home/yunxingl/imagenette2/examples", 
                        transform=RELATIVE_DISTANCE_TRANFORM)
        my_idx_to_class = dict(map(reversed, dataset.class_to_idx.items()))
        images = []
        labels = []
        for image, label in dataset: 
            image = image.unsqueeze(0)
            class_name = my_idx_to_class[label]
            label = class_to_idx[class_name]

            images.append(image)
            labels.append(label)
        self.images = torch.cat(images)
        self.labels = torch.Tensor(labels)
        
        self.num_classes = len(class_to_idx)

    def get_predicted_distance_label(self, embeddings, class_rep): 
        distances = torch.cdist(embeddings, class_rep) # num_data x (num_class * size of example dataloader)

        # reshape into num_data x num_class x size of example dataloader 
        dim_permutations = torch.argsort(self.labels)
        distances = distances[:, dim_permutations]
        distances = distances.reshape(len(distances), self.num_classes, -1)

        # take mean across size of example dataloader 
        mean_distances = distances.mean(dim=-1)
        # calculate mean across classes 
        return torch.argmax(mean_distances, dim=1)

    def get_predicted_sim_label(self, embeddings, class_rep): 
        r_norm = F.normalize(class_rep)
        e_norm = F.normalize(embeddings)
        sims = e_norm @ r_norm.t()

        dim_permutations = torch.argsort(self.labels)
        sims = sims[:, dim_permutations]
        sims = sims.reshape(len(sims), self.num_classes, -1)
        mean_sim = torch.mean(sims, dim=-1)
        return mean_sim, torch.argmax(mean_sim, dim=-1)

    def calculate_confusion_matrix(self, pred, actual): 
        pred = F.one_hot(pred, self.num_classes).bool()
        actual = F.one_hot(actual, self.num_classes).bool()
        pred = pred.unsqueeze(2)
        actual = actual.unsqueeze(1)
        eq = pred & actual 
        return eq.sum(dim=0)

    def update_confusion_matrix(self, pl_module, batch): 
        X, y = batch
        embeddings = pl_module.get_representation(X)
        projections = pl_module.get_projection(X)

        # get pairwise distances 
        zero_shot_rep = pl_module.get_representation(self.images)
        zero_shot_proj = pl_module.get_projection(self.images)
        dist_rep_label = self.get_predicted_distance_label(embeddings, zero_shot_rep)
        dist_proj_label = self.get_predicted_distance_label(projections, zero_shot_proj)
        self.rep_confusion_dist += self.calculate_confusion_matrix(dist_rep_label, y)
        self.proj_confusion_dist += self.calculate_confusion_matrix(dist_proj_label, y)

        # get max cosine similarities 
        mean_sim_rep, sim_rep_label = self.get_predicted_sim_label(embeddings, zero_shot_rep)
        mean_sim_proj, sim_proj_label = self.get_predicted_sim_label(projections, zero_shot_proj)
        self.rep_confusion_sim += self.calculate_confusion_matrix(sim_rep_label, y)
        self.proj_confusion_sim += self.calculate_confusion_matrix(sim_proj_label, y)
        self.proj_sim += update_similarity_table(mean_sim_proj, y, self.num_classes, pl_module.device)
        self.rep_sim += update_similarity_table(mean_sim_rep, y, self.num_classes, pl_module.device)


    def on_validation_start(self, trainer, pl_module): 
        self.images = self.images.to(pl_module.device)
        self.labels = self.labels.to(pl_module.device)
        self.rep_confusion_sim = self.rep_confusion_sim.to(pl_module.device)
        self.rep_confusion_dist = self.rep_confusion_dist.to(pl_module.device)
        self.proj_confusion_sim = self.proj_confusion_sim.to(pl_module.device)
        self.proj_confusion_dist = self.proj_confusion_dist.to(pl_module.device)
        self.rep_sim = self.rep_sim.to(pl_module.device)
        self.proj_sim = self.proj_sim.to(pl_module.device)
        print("moved matrices to devices")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.update_confusion_matrix(pl_module, batch)

    def save_data(self, output_path, additional_data): 
        data = {"rep_confusion_dist": self.rep_confusion_dist, 
                "rep_confusion_sim": self.rep_confusion_sim, 
                "proj_confusion_dist": self.proj_confusion_dist, 
                "proj_confusion_sim": self.proj_confusion_sim, 
                "sum_sim_proj": self.proj_sim, 
                "sum_sim_rep": self.rep_sim}
        data.update(additional_data)
        with open(output_path, 'wb') as handle: 
            pickle.dump(data, handle)
        self.reset_state()