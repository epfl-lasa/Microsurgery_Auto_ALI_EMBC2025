#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 16:41:46 2024

@author: luyin
"""

import yaml

def get_para(file_name):
    with open(file_name,'r') as f:
        data = yaml.safe_load(f)
    return data

#A = get_yaml('test.yaml')