from typing import List

import torch as ch
import torchvision
import os

from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter

def write_dataset(train_path, test_path, dest_path): 
    datasets = {
        'train': torchvision.datasets.ImageFolder(train_path),
        'test': torchvision.datasets.ImageFolder(test_path)
    }

    for (name, ds) in datasets.items():
        writer = DatasetWriter(os.path.join(dest_path, f"{name}_imagenette.beton"), {
            'image': RGBImageField(),
            'label': IntField()
        })
        writer.from_indexed_dataset(ds)

if __name__ == "__main__": 
    write_dataset("/mnt/nfs/home/yunxingl/imagenette2/train", 
                  "/mnt/nfs/home/yunxingl/imagenette2/val", 
                  "/mnt/nfs/datasets/imagenette")