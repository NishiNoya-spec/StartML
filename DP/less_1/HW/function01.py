import numpy as np

import torch
from torch import nn

def function01(tensor: torch.Tensor, count_over: str) -> torch.Tensor:

    if count_over == 'columns':
        return torch.mean(tensor, dim=0)
    elif count_over == 'rows':
        return torch.mean(tensor, dim=1)
    else:
        raise ValueError('No such value!')


if __name__ == '__main__':
    result = function01(
        tensor = torch.rand(3, 5),
        count_over = 'rows'
    )

    print(result)