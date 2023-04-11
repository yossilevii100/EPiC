#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Meir Yossef Levi
@Contact: me.levi@campus.technion.ac.il
@File: custom_model.py
@Time: 2023/04/11 6:35 PM
"""

import torch.nn as nn

class custom_model(nn.Module):
    def __init__(self, args, output_channels=40):
        super(custom_model, self).__init__()
        self.args = args
        self.output_channels = output_channels
        self.custom_linear = nn.Linear(64,4)

    def forward(self, x):
        raise NotImplementedError("Please implement your custom model")