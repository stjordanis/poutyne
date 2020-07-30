# pylint: disable=too-many-locals
import os
from unittest import TestCase
from unittest.mock import MagicMock, call, ANY

import numpy as np
import torch
import torch.nn as nn

from poutyne.framework import Callback


class ExperimentFittingTestCase(TestCase):
    epochs = 10
    steps_per_epoch = 5
    batch_size = 20

    evaluate_dataset_len = 107

    cuda_device = int(os.environ.get('CUDA_DEVICE', 0))

    def setUp(self):
        self.mock_callback = MagicMock(spec=Callback)
        self.batch_metrics = []
        self.batch_metrics_names = []
        self.batch_metrics_values = []
        self.epoch_metrics = []
        self.epoch_metrics_names = []
        self.epoch_metrics_values = []
        self.model = None
        self.pytorch_network = None




class MultiIOModel(nn.Module):
    """Model to test multiple inputs/outputs"""

    def __init__(self, num_input=2, num_output=2):
        super(MultiIOModel, self).__init__()
        inputs = []
        for _ in range(num_input):
            inputs.append(nn.Linear(1, 1))
        self.inputs = nn.ModuleList(inputs)

        outputs = []
        for _ in range(num_output):
            outputs.append(nn.Linear(num_input, 1))
        self.outputs = nn.ModuleList(outputs)

    def forward(self, *x):
        inp_to_cat = []
        for i, inp in enumerate(self.inputs):
            inp_to_cat.append(inp(x[i]))
        inp_cat = torch.cat(inp_to_cat, dim=1)

        outputs = []
        for out in self.outputs:
            outputs.append(out(inp_cat))

        outputs = outputs if len(outputs) > 1 else outputs[0]
        return outputs
