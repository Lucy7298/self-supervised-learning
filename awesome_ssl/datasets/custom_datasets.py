from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose
from torch.nn import Sequential
from typing import Union
import awesome_ssl.models.model_utils as model_utils
from typing import Sequence
from torchvision import transforms
from typing import Any, Callable, cast, Dict, List, Optional, Tuple


class SubsetFolder(ImageFolder): 
    def __init__(
        self,
        root: str,
        dirname_subset: List[str], 
        transform: Optional[Callable] = None,
        ) -> None:
        self.dirname_subset = dirname_subset
        super().__init__(root, transform=transform)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        classes = sorted(self.dirname_subset)
        class_to_idx = {cls_name: i for (i, cls_name) in enumerate(classes)}
        return classes, class_to_idx
    

class SimCLRAugDataset(ImageFolder):
  def __init__(self, 
               root : str, 
               image_size: int,
               transform_1: Union[Compose, Sequential], 
               transform_2: Union[Compose, Sequential], 
               debug: bool = False,
               *args, 
               **kwargs):
    super(SimCLRAugDataset, self).__init__(root)
    self.transform_1 = transform_1
    self.transform_2 = transform_2
    self.shared_transform = transforms.Compose([
      transforms.Resize(256),
      transforms.RandomResizedCrop(image_size),
      transforms.ToTensor(),
    ])
    self.debug = debug

  def __getitem__(self, index):
    # override ImageFolder's method
    """
    Args:
      index (int): Index
    Returns:
      tuple: (view_1, view_2, target) 
        view_1 is transform_1 version of the sample 
        view_2 is transform_2 version of the sample
        target is class_index of the target class.
    """
    path, target = self.samples[index]
    sample = self.loader(path)
    sample = self.shared_transform(sample)
    view_1 = self.transform_1(sample)
    view_2 = self.transform_2(sample)
    if self.target_transform is not None:
      target = self.target_transform(target)
    if not self.debug: 
      return view_1, view_2, target
    else: 
      return sample, view_1, view_2, target



class SimCLRAugDataset(ImageFolder):
  def __init__(self, 
               root : str, 
               image_size: int,
               transform_1: Union[Compose, Sequential], 
               transform_2: Union[Compose, Sequential], 
               debug: bool = False,
               *args, 
               **kwargs):
    super(SimCLRAugDataset, self).__init__(root)
    self.transform_1 = transform_1
    self.transform_2 = transform_2
    self.shared_transform = transforms.Compose([
      transforms.Resize(256),
      transforms.RandomResizedCrop(image_size),
      transforms.ToTensor(),
    ])
    self.debug = debug

  def __getitem__(self, index):
    # override ImageFolder's method
    """
    Args:
      index (int): Index
    Returns:
      tuple: (view_1, view_2, target) 
        view_1 is transform_1 version of the sample 
        view_2 is transform_2 version of the sample
        target is class_index of the target class.
    """
    path, target = self.samples[index]
    sample = self.loader(path)
    sample = self.shared_transform(sample)
    view_1 = self.transform_1(sample)
    view_2 = self.transform_2(sample)
    if self.target_transform is not None:
      target = self.target_transform(target)
    if not self.debug: 
      return view_1, view_2, target
    else: 
      return sample, view_1, view_2, target
