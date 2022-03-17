"""
Random grayscale 
"""
from dataclasses import replace
from numpy.random import rand
from typing import Callable, Optional, Tuple
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.state import State
from ffcv.pipeline.compiler import Compiler
import torch

class RandomGrayscale(Operation):
    """Randomly grayscales images. Only operates on torch image tensors 
    where each element is between [0, 1]

    Parameters
    ----------
    grayscale_prob : float
        The probability with which to grayscale each image in the batch
    """

    def __init__(self, grayscale_prob: float = 0.5):
        super().__init__()
        self.grayscale_prob = grayscale_prob

    def generate_code(self) -> Callable:
        my_range = Compiler.get_iterator()
        grayscale_prob = self.grayscale_prob

        def grayscale(images, dst):
            should_grayscale = rand(images.shape[0]) < grayscale_prob
            for i in my_range(images.shape[0]):
                if should_grayscale[i]:
                    dst[i] = images[i, 0:1, :, :] * 0.299 + \
                             images[i, 1:2, :, :] * 0.587 + images[i, 2:3, :, :] * 0.114
                else:
                    dst[i] = images[i]

            return dst

        grayscale.is_parallel = True
        return grayscale

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        return (replace(previous_state, jit_mode=True), 
                AllocationQuery(previous_state.shape, previous_state.dtype))

