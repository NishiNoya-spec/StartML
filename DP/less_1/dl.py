import numpy as np
import pandas as pd

import torch


if __name__ == "__main__":

    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name())

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(device)

